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

from model_service.builders.registry import get_builder
from model_service.facade.catalog import ModelCatalog
from model_service.providers.empaia.apps import (
    AppMode,
    EmpaiaListing,
    app_definitions,
    generate_ead,
)
from model_service.providers.empaia.credentials import ClientCredentials
from model_service.providers.empaia.registration import (
    Marketplace,
    register_until_done,
)
from model_service.providers.empaia.settings import EmpaiaSettings


MODELS = (
    Path(__file__).resolve().parents[2] / "deploy" / "helm" / "rayservice" / "models"
)


def catalog(*names: str) -> ModelCatalog:
    return ModelCatalog.from_documents(
        [yaml.safe_load((MODELS / f"{name}.yaml").read_text()) for name in names]
    )


class AppDefinitionTests(unittest.TestCase):
    def test_a_listed_model_offers_one_app_per_mode_of_each_publishable_builder(
        self,
    ) -> None:
        models = catalog("prostate-classifier-1", "virchow2")

        apps = app_definitions(models, models.empaia_listings)

        self.assertEqual(
            [(app.builder_id, app.model_id, app.mode) for app in apps],
            [
                ("heatmap", "prostate-classifier-1", AppMode.STANDALONE),
                ("heatmap", "prostate-classifier-1", AppMode.PREPROCESSING),
            ],
        )
        standalone, preprocessing = apps
        self.assertEqual(standalone.ead["name"], "Prostate cancer heatmap")
        self.assertEqual(
            standalone.ead["modes"],
            {
                "standalone": {
                    "inputs": ["slide", "region_of_interest"],
                    "outputs": ["heatmap"],
                }
            },
        )
        self.assertEqual(
            standalone.ead["io"]["region_of_interest"]["reference"], "io.slide"
        )
        # EMPAIA starts a preprocessing app with a slide alone.
        self.assertEqual(
            preprocessing.ead["modes"],
            {"preprocessing": {"inputs": ["slide"], "outputs": ["heatmap"]}},
        )
        self.assertNotEqual(standalone.app_id, preprocessing.app_id)
        self.assertEqual(
            standalone.image("model-service"),
            "http::model-service/builders/heatmap/models/prostate-classifier-1",
        )

    def test_every_listed_model_in_the_repository_yields_valid_apps(self) -> None:
        listed = 0
        for path in sorted(MODELS.rglob("*.yaml")):
            with self.subTest(model=str(path.relative_to(MODELS))):
                models = catalog(str(path.relative_to(MODELS).with_suffix("")))
                apps = app_definitions(models, models.empaia_listings)
                if models.empaia_listings:
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
        spec = get_builder("heatmap")
        heatmap = spec.outputs["heatmap"]
        broken = replace(
            spec,
            outputs={
                "heatmap": replace(
                    heatmap, empaia={**heatmap.empaia, "type": "no_such_type"}
                )
            },
        )
        with self.assertRaises(ValidationError):
            generate_ead(broken, "model", listing, AppMode.STANDALONE)


def token(lifetime: float) -> str:
    now = time.time()
    return jwt.encode(
        {"iat": now, "exp": now + lifetime}, "a-key-nobody-checks-in-this-test-1234"
    )


class RegistrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_apps_are_posted_with_client_credentials(self) -> None:
        posted: list[tuple[str, dict[str, object]]] = []
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
            posted.append(
                (request.headers["authorization"], json.loads(request.content))
            )
            return httpx.Response(201)

        models = catalog("prostate-classifier-1")
        apps = app_definitions(models, models.empaia_listings)
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

        self.assertEqual(len(posted), 4)
        self.assertTrue(all(header.startswith("Bearer ") for header, _ in posted))
        self.assertEqual(tokens, 1, "a valid token is reused")
        app = posted[0][1]["active_app_views"]["v3"]["app"]  # type: ignore[index]
        self.assertEqual(app["id"], apps[0].app_id)
        self.assertEqual(app["registry_image_url"], apps[0].image("model-service"))

    async def test_registration_is_retried_until_the_marketplace_answers(self) -> None:
        answers = [httpx.Response(503), httpx.Response(201), httpx.Response(201)]

        def handle(request: httpx.Request) -> httpx.Response:
            self.assertNotIn("authorization", request.headers)
            return answers.pop(0)

        models = catalog("prostate-classifier-1")
        apps = app_definitions(models, models.empaia_listings)
        settings = EmpaiaSettings(
            marketplace_url="https://marketplace.example", organization_id="org-1"
        )
        from model_service.facade.app import _marketplace

        marketplace = _marketplace(settings)
        assert marketplace is not None
        marketplace._transport = httpx.MockTransport(handle)

        await register_until_done(marketplace, apps, retry_seconds=0)

        self.assertEqual(answers, [])

    def test_a_marketplace_needs_an_organization(self) -> None:
        with self.assertRaises(SettingsError):
            EmpaiaSettings(marketplace_url="https://marketplace.example")
        with self.assertRaises(SettingsError):
            EmpaiaSettings(idp_url="https://idp.example")


if __name__ == "__main__":
    unittest.main()
