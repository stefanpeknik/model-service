"""Analyses with `expose.empaia` become EMPAIA apps, registered in the marketplace at startup."""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from pathlib import Path

import httpx
import jwt
import pytest
from pydantic import ValidationError as SettingsError

from model_service.builders.registry import BUILDERS
from model_service.catalog import load_catalog
from model_service.providers.empaia.credentials import (
    ClientCredentials,
    NoCredentials,
)
from model_service.providers.empaia.ead.apps import (
    AnalysisApp,
    analysis_app,
    analysis_apps,
)
from model_service.providers.empaia.ead.builder_io import BUILDER_IO, NOT_OFFERED
from model_service.providers.empaia.ead.vocabulary import VIEWER_CONFIG_KEY, AppMode
from model_service.providers.empaia.registration import (
    Marketplace,
    register_until_done,
)
from model_service.providers.empaia.settings import EmpaiaSettings
from model_service.providers.empaia.triggers import PreprocessingTriggers
from tests.support.analysis_catalog import REGISTRY, write
from tests.support.analysis_catalog import served_model as model


CATALOG = Path(__file__).resolve().parents[2] / "catalog"
# Today's app, as the builder-keyed EAD generation made it from
# prostate-classifier-1's `empaia` block (recorded at e91f62d).
BEFORE = json.loads(
    (Path(__file__).parent / "fixtures/prostate-heatmap-app-before.json").read_text()
)
OLD_NAMESPACE = "org.empaia.rationai.heatmap_prostate_classifier_1.v3.0"
NEW_NAMESPACE = "org.empaia.rationai.prostate_heatmap.v3.0"


def prostate_app(environment: str | None = None) -> AnalysisApp:
    analyses = load_catalog(CATALOG, environment).entries("analyses")
    return analysis_app(analyses["prostate-heatmap"].value)


def tmp_app(tmp_path: Path, expose: str) -> AnalysisApp:
    """The app of a small analysis with this `expose.empaia` body."""
    root = tmp_path / "catalog"
    write(root, "served-models/m1.yaml", model())
    write(
        root,
        "analyses/a1.yaml",
        """
        source: {use: slide_tiles}
        steps: [{model: m1}]
        sinks: [{use: score_grid, output: heatmap}]
        expose:
          empaia:
            name: Test heatmap
            name_short: TestHM
            description: Finds things.
            outputs:
              heatmap: {type: continuous_pixelmap}
        """.rstrip(" ")
        + "\n".join(f"            {line}" for line in expose.splitlines()),
    )
    analyses = load_catalog(root, registry=REGISTRY).entries("analyses")
    return analysis_app(analyses["a1"].value)


@pytest.mark.parametrize("environment", [None, "rationai-test"])
def test_the_prostate_app_is_todays_apart_from_its_namespace(
    environment: str | None,
) -> None:
    app = prostate_app(environment)

    assert app.ead["namespace"] == NEW_NAMESPACE
    assert {**app.ead, "namespace": OLD_NAMESPACE} == BEFORE["ead"]
    assert app.global_configuration == BEFORE["global_configuration"]
    assert app.modes == (AppMode.STANDALONE, AppMode.PREPROCESSING)
    assert app.image("model-service") == (
        "http::model-service/analyses/prostate-heatmap"
    )
    # Its triggers' tags list the app under them in the marketplace.
    tags = app.portal_app("model-service", "org-1")["active_app_views"]["v3"]["tags"]
    assert (tags["tissues"][0]["name"], tags["stains"][0]["name"]) == (
        "PROSTATE",
        "H_AND_E",
    )


def test_every_exposed_analysis_yields_a_valid_app_in_every_environment() -> None:
    for environment in (None, "rationai-test"):
        apps = analysis_apps(load_catalog(CATALOG, environment))

        assert [app.analysis_id for app in apps] == ["prostate-heatmap"]


def test_a_preprocessing_only_app_leaves_out_the_region_of_interest(
    tmp_path: Path,
) -> None:
    app = tmp_app(tmp_path, "modes: {preprocessing: }")

    assert "region_of_interest" not in app.ead["io"]
    assert app.ead["modes"] == {
        "preprocessing": {"inputs": ["slide"], "outputs": ["heatmap"]}
    }
    assert "configuration" not in app.ead


def test_parameters_become_standalone_inputs_described_by_their_settings(
    tmp_path: Path,
) -> None:
    app = tmp_app(
        tmp_path,
        "modes:\n"
        "  standalone: {parameters: [min_tissue_fraction, skip_background]}\n"
        "  preprocessing:",
    )

    assert app.ead["io"]["min_tissue_fraction"]["type"] == "float"
    assert app.ead["io"]["min_tissue_fraction"]["description"].startswith(
        "How much of a tile must be tissue"
    )
    assert app.ead["io"]["skip_background"]["type"] == "bool"
    # EMPAIA starts preprocessing with the slide alone, so it runs on the file's values.
    assert app.ead["modes"]["standalone"]["inputs"] == [
        "slide",
        "region_of_interest",
        "min_tissue_fraction",
        "skip_background",
    ]
    assert app.ead["modes"]["preprocessing"]["inputs"] == ["slide"]
    # The case browser may pre-fill its form from the file's values.
    assert VIEWER_CONFIG_KEY in app.ead["configuration"]["global"]
    viewer = json.loads(app.global_configuration[VIEWER_CONFIG_KEY])
    assert viewer["inputs"] == {
        "min_tissue_fraction": {"default": 0.0},
        "skip_background": {"default": True},
    }


def test_viewer_entries_for_io_an_app_lacks_are_left_out(tmp_path: Path) -> None:
    app = tmp_app(
        tmp_path,
        "modes: {preprocessing: }\n"
        "viewer:\n"
        "  inputs: {region_of_interest: {label: ROI}}\n"
        "  outputs: {heatmap: {render: shader}}",
    )

    assert json.loads(app.global_configuration[VIEWER_CONFIG_KEY]) == {
        "inputs": {},
        "outputs": {"heatmap": {"render": "shader"}},
    }


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

        apps = [prostate_app()]
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

        apps = [prostate_app()]
        settings = EmpaiaSettings(
            marketplace_url="https://marketplace.example", organization_id="org-1"
        )
        from model_service.providers.empaia.provider import (
            _credentials,
            _marketplace,
            _triggers,
        )

        credentials = _credentials(settings)
        marketplace = _marketplace(settings, credentials)
        assert marketplace is not None
        self.assertIsNone(_triggers(settings, credentials), "no MDS, no triggers")
        marketplace._transport = httpx.MockTransport(handle)

        with self.assertLogs("model_service.providers.empaia.registration") as logs:
            await register_until_done(marketplace, apps, retry_seconds=0)

        self.assertEqual(answers, [])
        # Repeated failures are a setting, not an outage: errors from the third.
        self.assertEqual(
            [r.levelname for r in logs.records][:3], ["WARNING", "WARNING", "ERROR"]
        )

    def test_a_medical_data_service_needs_a_marketplace(self) -> None:
        with self.assertRaises(SettingsError):
            EmpaiaSettings(mds_url="http://mds.example")


class TriggerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.app = prostate_app()
        self.items: list[dict[str, str]] = []
        self.created = 0

    def mds(self, owner: str) -> PreprocessingTriggers:
        """A client for an in-memory Medical Data Service shared by every client."""

        async def handle(request: httpx.Request) -> httpx.Response:
            # Let other clients' requests interleave, as against a real service.
            await asyncio.sleep(0)
            if request.method == "GET":
                return httpx.Response(200, json={"items": list(self.items)})
            if request.method == "POST":
                self.created += 1
                body = json.loads(request.content)
                self.items.append({"id": f"n{self.created}", **body})
                return httpx.Response(201)
            trigger_id = request.url.path.rsplit("/", 1)[1]
            if all(item["id"] != trigger_id for item in self.items):
                return httpx.Response(404)
            self.items[:] = [i for i in self.items if i["id"] != trigger_id]
            return httpx.Response(200)

        return PreprocessingTriggers(
            "http://mds.example",
            owner,
            NoCredentials(),
            transport=httpx.MockTransport(handle),
        )

    def trigger(self, trigger_id: str, owner: str, tissue: str) -> dict[str, str]:
        return {
            "id": trigger_id,
            "creator_id": owner,
            "creator_type": "USER",
            "portal_app_id": self.app.portal_app_id,
            "tissue": tissue,
            "stain": "H_AND_E",
        }

    async def test_owned_triggers_follow_the_apps_and_others_are_left_alone(
        self,
    ) -> None:
        ours = "model-service:model-service"
        self.items[:] = [
            # Ours for the listed pair comes first in the list ...
            self.trigger("a", ours, "PROSTATE"),
            # ... yet someone else's already covers it, so ours goes.
            self.trigger("b", "admin", "PROSTATE"),
            # Ours, no longer listed.
            self.trigger("c", ours, "BREAST"),
            # Someone else's, not listed: never touched.
            self.trigger("d", "admin", "LUNG"),
        ]

        await self.mds(ours).ensure([self.app])

        self.assertEqual([item["id"] for item in self.items], ["b", "d"])

        await self.mds(ours).ensure([])
        self.assertEqual([item["id"] for item in self.items], ["b", "d"])

    async def test_replicas_starting_together_end_with_one_trigger(self) -> None:
        ours = "model-service:model-service"
        first, second = self.mds(ours), self.mds(ours)

        await asyncio.gather(first.ensure([self.app]), second.ensure([self.app]))

        self.assertEqual(len(self.items), 1)
        self.assertEqual(
            {k: v for k, v in self.items[0].items() if k != "id"},
            {k: v for k, v in self.trigger("", ours, "PROSTATE").items() if k != "id"},
        )


class BuilderCoverageTests(unittest.TestCase):
    def test_every_builder_is_offered_or_explicitly_not(self) -> None:
        self.assertEqual(BUILDER_IO.keys() & NOT_OFFERED.keys(), set())
        self.assertEqual(BUILDER_IO.keys() | NOT_OFFERED.keys(), BUILDERS.keys())


if __name__ == "__main__":
    unittest.main()
