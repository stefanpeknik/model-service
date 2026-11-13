"""Deployed models become EMPAIA apps, registered in the marketplace at startup."""

from __future__ import annotations

import json
import time
import unittest
from dataclasses import replace
from pathlib import Path

import httpx
import jwt
import yaml
from jsonschema import ValidationError
from pydantic import ValidationError as SettingsError

from model_service.builders.registry import BUILDERS, get_builder
from model_service.deployed_models.catalog import ModelCatalog
from model_service.providers.empaia.credentials import ClientCredentials
from model_service.providers.empaia.ead.apps import (
    VIEWER_CONFIG_KEY,
    AppMode,
    EmpaiaListing,
    ViewerConfig,
    app_definitions,
    generate_ead,
    parse_listings,
)
from model_service.providers.empaia.ead.builder_io import BUILDER_IO, NOT_OFFERED
from model_service.providers.empaia.provider import CATALOG_BLOCK
from model_service.providers.empaia.publishing.publishers import default_publishers
from model_service.providers.empaia.registration import (
    Marketplace,
    register_until_done,
)
from model_service.providers.empaia.settings import EmpaiaSettings
from tests.support.deployment import CHART, VALUES, load


PUBLISHERS = default_publishers()
MODELS = (
    Path(__file__).resolve().parents[2] / "deploy" / "helm" / "rayservice" / "models"
)


def catalog(*names: str) -> ModelCatalog:
    return ModelCatalog.from_documents(
        [yaml.safe_load((MODELS / f"{name}.yaml").read_text()) for name in names],
        [CATALOG_BLOCK],
    )


def listings(models: ModelCatalog) -> dict[str, EmpaiaListing]:
    return parse_listings(models.provider_block(CATALOG_BLOCK))


class AppDefinitionTests(unittest.TestCase):
    def test_a_listed_model_offers_one_app_with_its_modes_per_builder(
        self,
    ) -> None:
        models = catalog("prostate-classifier-1", "virchow2")

        (app,) = app_definitions(models, listings(models), PUBLISHERS)

        self.assertEqual(
            (app.builder_id, app.model_id, app.modes),
            (
                "heatmap",
                "prostate-classifier-1",
                (AppMode.STANDALONE, AppMode.PREPROCESSING),
            ),
        )
        self.assertEqual(app.ead["name"], "Prostate cancer heatmap")
        self.assertEqual(
            app.ead["namespace"],
            "org.empaia.rationai.heatmap_prostate_classifier_1.v3.0",
        )
        # EMPAIA starts a preprocessing job with a slide alone.
        self.assertEqual(
            app.ead["modes"],
            {
                "standalone": {
                    "inputs": ["slide", "region_of_interest"],
                    "outputs": ["heatmap"],
                },
                "preprocessing": {"inputs": ["slide"], "outputs": ["heatmap"]},
            },
        )
        self.assertEqual(app.ead["io"]["region_of_interest"]["reference"], "io.slide")
        self.assertEqual(
            app.image("model-service"),
            "http::model-service/builders/heatmap/models/prostate-classifier-1",
        )

    def test_a_preprocessing_only_app_leaves_out_the_region_of_interest(self) -> None:
        listing = EmpaiaListing(
            name="Prostate cancer",
            name_short="ProstHM",
            description="Finds things.",
            modes=(AppMode.PREPROCESSING,),
        )
        models = catalog("prostate-classifier-1")

        (app,) = app_definitions(models, {"prostate-classifier-1": listing}, PUBLISHERS)

        self.assertNotIn("region_of_interest", app.ead["io"])
        self.assertEqual(list(app.ead["modes"]), ["preprocessing"])

    def test_the_viewer_settings_are_declared_and_given_to_each_app(self) -> None:
        models = catalog("prostate-classifier-1")

        (standalone,) = app_definitions(models, listings(models), PUBLISHERS)

        self.assertEqual(
            standalone.ead["configuration"]["global"][VIEWER_CONFIG_KEY]["type"],
            "string",
        )
        viewer = json.loads(standalone.global_configuration[VIEWER_CONFIG_KEY])
        self.assertEqual(
            viewer["inputs"]["slide"], {"target": "xopat", "lossless": False}
        )
        self.assertEqual(viewer["outputs"]["heatmap"]["shader"]["type"], "heatmap")

    def test_an_invalid_listing_leaves_only_its_model_out(self) -> None:
        valid = {
            "name": "Prostate",
            "name_short": "Pros",
            "description": "Finds.",
            "modes": ["preprocessing"],
        }
        with self.assertLogs(
            "model_service.providers.empaia.ead.apps", "ERROR"
        ) as logs:
            parsed = parse_listings(
                {
                    "good": valid,
                    "typo": {**valid, "mods": ["preprocessing"]},
                    "unknown-builder": {**valid, "viewer": {"heatmapp": {}}},
                }
            )
        self.assertEqual(list(parsed), ["good"])
        self.assertEqual(len(logs.output), 2)

    def test_viewer_entries_for_io_an_app_lacks_are_left_out(self) -> None:
        listing = EmpaiaListing(
            name="Prostate cancer",
            name_short="ProstHM",
            description="Finds things.",
            modes=(AppMode.PREPROCESSING,),
            viewer={
                "heatmap": ViewerConfig.model_validate(
                    {
                        "inputs": {"region_of_interest": {"label": "ROI"}},
                        "outputs": {"heatmap": {"render": "shader"}},
                    }
                )
            },
        )
        models = catalog("prostate-classifier-1")

        (app,) = app_definitions(
            models,
            {"prostate-classifier-1": listing},
            PUBLISHERS,
            [get_builder("heatmap")],
        )

        self.assertEqual(
            json.loads(app.global_configuration[VIEWER_CONFIG_KEY]),
            {"inputs": {}, "outputs": {"heatmap": {"render": "shader"}}},
        )

    def test_every_listed_model_in_the_repository_yields_valid_apps(self) -> None:
        listed = 0
        for path in sorted(MODELS.rglob("*.yaml")):
            with self.subTest(model=str(path.relative_to(MODELS))):
                models = catalog(str(path.relative_to(MODELS).with_suffix("")))
                apps = app_definitions(models, listings(models), PUBLISHERS)
                if listings(models):
                    listed += 1
                    self.assertTrue(apps)
        self.assertGreater(listed, 0)

    def test_listings_keep_within_the_ead_limits(self) -> None:
        with self.assertRaises(SettingsError):
            EmpaiaListing(
                name="A model name far too long",
                name_short="Model",
                description="Finds things.",
                modes=(AppMode.STANDALONE,),
            )
        listing = EmpaiaListing(
            name="Prostate cancer",
            name_short="ProstHM",
            description="Finds things.",
            modes=(AppMode.STANDALONE,),
        )
        heatmap = BUILDER_IO["heatmap"]
        broken = replace(heatmap, outputs={"heatmap": {"type": "no_such_type"}})
        with self.assertRaises(ValidationError):
            generate_ead(get_builder("heatmap"), broken, "model", listing)


def token(lifetime: float) -> str:
    now = time.time()
    return jwt.encode(
        {"iat": now, "exp": now + lifetime}, "a-key-nobody-checks-in-this-test-1234"
    )


class RegistrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_apps_are_posted_with_client_credentials(self) -> None:
        posted: list[tuple[str, dict[str, object]]] = []
        configured: list[tuple[str, dict[str, object]]] = []
        tokens = 0

        def handle(request: httpx.Request) -> httpx.Response:
            nonlocal tokens
            if request.url.path.endswith("/openid-configuration"):
                return httpx.Response(
                    200, json={"token_endpoint": "https://idp.example/token"}
                )
            if request.url.path == "/token":
                tokens += 1
                return httpx.Response(200, json={"access_token": token(600)})
            sent = (request.headers["authorization"], json.loads(request.content))
            if request.method == "PUT":
                configured.append((request.url.path, sent[1]))
                return httpx.Response(200)
            posted.append(sent)
            return httpx.Response(201)

        models = catalog("prostate-classifier-1")
        apps = app_definitions(models, listings(models), PUBLISHERS)
        credentials = ClientCredentials("https://idp.example/realm", "ms", "secret")
        marketplace = Marketplace(
            "https://marketplace.example",
            "org-1",
            "model-service",
            credentials,
            transport=httpx.MockTransport(handle),
        )

        await marketplace.register(apps)
        await marketplace.register(apps)

        self.assertEqual(len(posted), 2)
        self.assertTrue(all(header.startswith("Bearer ") for header, _ in posted))
        self.assertEqual(tokens, 1, "a valid token is reused")
        app = posted[0][1]["active_app_views"]["v3"]["app"]  # type: ignore[index]
        self.assertEqual(app["id"], apps[0].app_id)
        self.assertEqual(app["registry_image_url"], apps[0].image("model-service"))
        self.assertEqual(
            configured[0],
            (
                f"/v1/custom-mock/apps/{apps[0].app_id}/config/global",
                apps[0].global_configuration,
            ),
        )

    async def test_registration_is_retried_until_the_marketplace_answers(self) -> None:
        # The app: the portal app, then its viewer configuration.
        answers = [
            httpx.Response(503),
            httpx.Response(401),
            httpx.Response(401),
            httpx.Response(201),
            httpx.Response(200),
        ]

        def handle(request: httpx.Request) -> httpx.Response:
            self.assertNotIn("authorization", request.headers)
            return answers.pop(0)

        models = catalog("prostate-classifier-1")
        apps = app_definitions(models, listings(models), PUBLISHERS)
        settings = EmpaiaSettings(
            marketplace_url="https://marketplace.example", organization_id="org-1"
        )
        from model_service.providers.empaia.provider import _marketplace

        marketplace = _marketplace(settings)
        assert marketplace is not None
        marketplace._transport = httpx.MockTransport(handle)

        with self.assertLogs("model_service.providers.empaia.registration") as logs:
            await register_until_done(marketplace, apps, retry_seconds=0)

        self.assertEqual(answers, [])
        # Repeated failures are a setting, not an outage: errors from the third.
        self.assertEqual(
            [r.levelname for r in logs.records][:3], ["WARNING", "WARNING", "ERROR"]
        )
        self.assertIn("HTTP 401 from https://marketplace.example/", logs.output[2])


class BuilderCoverageTests(unittest.TestCase):
    def test_every_builder_is_offered_or_explicitly_not(self) -> None:
        self.assertEqual(BUILDER_IO.keys() & NOT_OFFERED.keys(), set())
        self.assertEqual(BUILDER_IO.keys() | NOT_OFFERED.keys(), BUILDERS.keys())


class ValuesFileTests(unittest.TestCase):
    def test_each_values_file_yields_a_valid_catalog_and_its_apps(self) -> None:
        publishers = default_publishers()
        for values in VALUES:
            models = load(values).get("models")
            if not models:
                continue
            with self.subTest(values.name):
                documents = [load(CHART / f"models/{m}.yaml") for m in models]
                catalog = ModelCatalog.from_documents(documents, [CATALOG_BLOCK])
                listings = parse_listings(catalog.provider_block(CATALOG_BLOCK))
                apps = app_definitions(catalog, listings, publishers)
                # A model listed for EMPAIA that no builder can serve there would
                # otherwise only be logged at startup.
                offered = {app.model_id for app in apps}
                self.assertEqual(offered, set(listings))


if __name__ == "__main__":
    unittest.main()
