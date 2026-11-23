"""Analyses with `expose.empaia` become EMPAIA apps, registered in the marketplace at startup."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

import httpx
import jwt
import pytest
from pydantic import ValidationError as SettingsError

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
# The prostate app as the test marketplace listed it at e91f62d, under its
# earlier namespace.
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
def test_the_prostate_app_matches_its_recording_apart_from_its_namespace(
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
        catalog = load_catalog(CATALOG, environment)
        apps = analysis_apps(catalog)

        exposed = [i for i, e in catalog.entries("analyses").items() if e.value.empaia]
        assert [app.analysis_id for app in apps] == exposed
        assert "prostate-heatmap" in exposed


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
        # The app: the portal app, then its viewer configuration, then the
        # marketplace's apps, none to retire.
        answers = [
            httpx.Response(503),
            httpx.Response(401),
            httpx.Response(401),
            httpx.Response(201),
            httpx.Response(200),
            httpx.Response(200, json={"items": []}),
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


def listed_app(portal_app_id: str, namespace: str, image: str) -> dict[str, object]:
    """A portal app as the marketplace mock lists it, trimmed to what owning reads."""
    return {
        "id": portal_app_id,
        "active_app_views": {
            "v1": None,
            "v3": {
                "app": {
                    "id": f"app-of-{portal_app_id}",
                    "ead": {"namespace": namespace},
                    "registry_image_url": image,
                }
            },
        },
    }


# An app this deployment registered under the prostate app's earlier
# namespace, at a route no analysis has now.
RETIRED_APP = listed_app(
    "retired",
    OLD_NAMESPACE,
    "http::model-service/analyses/heatmap-prostate-classifier-1",
)
# Apps model-service never registered: a RationAI container app under the same
# namespace prefix, another model-service deployment's app, another vendor's.
FOREIGN_APPS = [
    listed_app(
        "container",
        "org.empaia.rationai.gigapath_embed_score.v3.0",
        "registry.example/rationai/gigapath-embed-score:1.0",
    ),
    listed_app(
        "other-deployment",
        "org.empaia.rationai.constant.v3.0",
        "http::model-service-dev/analyses/constant",
    ),
    listed_app(
        "other-vendor", "org.empaia.vendor.app.v3.0", "registry.example/vendor/app:1"
    ),
]


class FakeMarketplace:
    """The marketplace mock's portal-app routes, in memory, shared by every client."""

    def __init__(self, *apps: dict[str, object]) -> None:
        self.apps = {str(app["id"]): app for app in apps}
        self.deleted: list[str] = []

    def client(self) -> Marketplace:
        async def handle(request: httpx.Request) -> httpx.Response:
            # Let other clients' requests interleave, as against a real service.
            await asyncio.sleep(0)
            path = request.url.path.removeprefix("/v1")
            if request.method == "POST" and path == "/custom-mock/portal-apps":
                body = json.loads(request.content)
                self.apps[body["id"]] = body
                return httpx.Response(200)
            if request.method == "PUT" and path.startswith("/custom-mock/apps/"):
                return httpx.Response(200)
            if request.method == "PUT" and path == "/customer/portal-apps/query":
                assert request.headers["organization-id"] == "org-1"
                items = list(self.apps.values())
                return httpx.Response(
                    200, json={"item_count": len(items), "items": items}
                )
            if request.method == "DELETE" and path.startswith(
                "/custom-mock/portal-apps/"
            ):
                portal_app_id = path.rsplit("/", 1)[1]
                if self.apps.pop(portal_app_id, None) is None:
                    return httpx.Response(404)
                self.deleted.append(portal_app_id)
                return httpx.Response(200)
            return httpx.Response(405)

        return Marketplace(
            "https://marketplace.example",
            "org-1",
            "model-service",
            NoCredentials(),
            transport=httpx.MockTransport(handle),
        )


class ReconcileTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_renamed_app_replaces_the_old_and_foreign_apps_stay(
        self,
    ) -> None:
        marketplace = FakeMarketplace(RETIRED_APP, *FOREIGN_APPS)
        app = prostate_app()

        await register_until_done(marketplace.client(), [app], retry_seconds=0)

        self.assertEqual(
            sorted(marketplace.apps),
            sorted([app.portal_app_id, *(a["id"] for a in FOREIGN_APPS)]),
        )
        self.assertEqual(marketplace.deleted, ["retired"])

    async def test_replicas_reconciling_together_and_again_agree(self) -> None:
        marketplace = FakeMarketplace(RETIRED_APP, *FOREIGN_APPS)
        app = prostate_app()
        expected = sorted([app.portal_app_id, *(a["id"] for a in FOREIGN_APPS)])

        # Both retire the old app; neither takes the other's success as a failure.
        with self.assertNoLogs(
            "model_service.providers.empaia.registration", level="WARNING"
        ):
            await asyncio.gather(
                register_until_done(marketplace.client(), [app], retry_seconds=0),
                register_until_done(marketplace.client(), [app], retry_seconds=0),
            )
        self.assertEqual(sorted(marketplace.apps), expected)

        # A replica restarting later replaces the app with itself, nothing more.
        await register_until_done(marketplace.client(), [app], retry_seconds=0)

        self.assertEqual(sorted(marketplace.apps), expected)
        self.assertEqual(marketplace.deleted, ["retired"])
        registered = marketplace.apps[app.portal_app_id]["active_app_views"]["v3"]
        self.assertEqual(
            (registered["app"]["id"], registered["app"]["ead"]), (app.app_id, app.ead)
        )


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

    async def test_each_analysis_triggers_its_own_app_only(self) -> None:
        ours = "model-service:model-service"
        # Ours, for the retired app the prostate analysis replaces.
        self.items[:] = [
            {**self.trigger("a", ours, "PROSTATE"), "portal_app_id": "retired"}
        ]
        with tempfile.TemporaryDirectory() as tmp:
            other = tmp_app(
                Path(tmp),
                "modes:\n"
                "  preprocessing:\n"
                "    triggers: [{tissue: PROSTATE, stain: H_AND_E}]",
            )
        marketplace = FakeMarketplace(RETIRED_APP)

        await register_until_done(
            marketplace.client(), [self.app, other], self.mds(ours), retry_seconds=0
        )

        self.assertEqual(
            sorted((item["portal_app_id"], item["tissue"]) for item in self.items),
            sorted(
                [
                    (self.app.portal_app_id, "PROSTATE"),
                    (other.portal_app_id, "PROSTATE"),
                ]
            ),
        )
        self.assertEqual(
            sorted(marketplace.apps),
            sorted([self.app.portal_app_id, other.portal_app_id]),
        )

    async def test_replicas_starting_together_end_with_one_trigger(self) -> None:
        ours = "model-service:model-service"
        first, second = self.mds(ours), self.mds(ours)

        await asyncio.gather(first.ensure([self.app]), second.ensure([self.app]))

        self.assertEqual(len(self.items), 1)
        self.assertEqual(
            {k: v for k, v in self.items[0].items() if k != "id"},
            {k: v for k, v in self.trigger("", ours, "PROSTATE").items() if k != "id"},
        )


if __name__ == "__main__":
    unittest.main()
