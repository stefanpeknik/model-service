"""The EMPAIA app of each analysis with `expose.empaia`, and its EAD.

The EAD merges what each input's converter and each output's publisher
contribute (their `io` entries) with the namespace, which comes from the
analysis ID, the modes, the permissions and the viewer configuration key.
Standalone mode takes every input and the parameters; preprocessing only
the required inputs, because EMPAIA starts it with a slide alone. The app's
image is `http::<target>/analyses/<id>`, which JES hands to model-service.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, UUID, uuid5

from jsonschema import Draft7Validator

from components.base import EmpaiaInput, EmpaiaOutput
from model_service.providers.empaia.ead.vocabulary import (
    VIEWER_CONFIG_KEY,
    AppMode,
    Trigger,
)
from model_service.providers.empaia.tags import ANY, STAINS, TISSUES


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from model_service.catalog import Catalog
    from model_service.catalog.analyses import Analysis


EAD_SCHEMA_URL = "https://gitlab.com/empaia/integration/definitions/-/raw/main/ead/ead-schema.v3.json"
_EAD_SCHEMA = Draft7Validator(
    json.loads(Path(__file__).with_name("ead-v3.schema.json").read_text())
)
NAMESPACE_PREFIX = "org.empaia.rationai."


@dataclass(frozen=True, slots=True)
class AnalysisApp:
    """The EMPAIA app of one analysis.

    Attributes:
        analysis_id: The analysis it runs.
        modes: The job modes it supports.
        ead: Its EAD.
        global_configuration: Values for the EAD's `configuration.global`,
            set when the app is registered.
        triggers: The slides EMPAIA preprocesses with it.
    """

    analysis_id: str
    modes: tuple[AppMode, ...]
    ead: dict[str, Any]
    global_configuration: dict[str, str] = field(default_factory=dict)
    triggers: tuple[Trigger, ...] = ()

    def image(self, target: str) -> str:
        """Build the app's image reference, which JES hands to model-service.

        Args:
            target: model-service's name in JES's HTTP targets.

        Returns:
            `http::<target>/analyses/<analysis id>`.
        """
        return f"http::{target}/analyses/{self.analysis_id}"

    @property
    def app_id(self) -> str:
        """The ID EMPAIA jobs name the app by."""
        return stable_id("empaia-app", self.ead["namespace"])

    @property
    def portal_app_id(self) -> str:
        """The ID the marketplace lists the app by, and triggers name it by."""
        return stable_id("empaia-portal-app", self.ead["namespace"])

    def portal_app(self, target: str, organization_id: str) -> dict[str, Any]:
        """Build the marketplace mock's portal app.

        Its IDs derive from the namespace, so registering it again replaces it.

        Args:
            target: model-service's name in JES's HTTP targets, part of the image.
            organization_id: The organization that creates, owns and reviews the app.

        Returns:
            The portal app body: listed, approved, with the EAD and image.
        """
        namespace = self.ead["namespace"]
        version = namespace[namespace.rfind(".") - 2 :]
        now = int(time.time())
        common = {
            "portal_app_id": self.portal_app_id,
            "creator_id": organization_id,
            "created_at": now,
        }
        return {
            "id": common["portal_app_id"],
            "organization_id": organization_id,
            "status": "LISTED",
            "creator_id": organization_id,
            "created_at": now,
            "updated_at": now,
            "active_app_views": {
                "v3": {
                    **common,
                    "organization_id": organization_id,
                    "id": stable_id("empaia-app-view", namespace),
                    "version": version,
                    "api_version": "v3",
                    "status": "APPROVED",
                    "non_functional": False,
                    "research_only": True,
                    "details": {
                        "name": self.ead["name"],
                        "marketplace_url": "http://url.to/store",
                        "description": [
                            {"lang": "EN", "text": self.ead["description"]}
                        ],
                    },
                    "media": {"peek": [], "banner": [], "workflow": [], "manual": []},
                    "tags": {
                        "tissues": _tags(TISSUES, (t.tissue for t in self.triggers)),
                        "stains": _tags(STAINS, (t.stain for t in self.triggers)),
                        "indications": [],
                        "analysis": [],
                        "clearances": [],
                    },
                    "review_comment": "registered by model-service",
                    "reviewer_id": organization_id,
                    "reviewed_at": now,
                    "app": {
                        **common,
                        "id": self.app_id,
                        "ead": self.ead,
                        "registry_image_url": self.image(target),
                        "app_ui_url": None,
                        "version": version,
                        "status": "APPROVED",
                        "has_frontend": False,
                        "updated_at": now,
                    },
                }
            },
        }


def analysis_apps(catalog: Catalog) -> list[AnalysisApp]:
    """The app of every deployed analysis with `expose.empaia`.

    Args:
        catalog: The deployed catalog.

    Returns:
        One app per exposed analysis, by analysis ID.

    Raises:
        jsonschema.ValidationError: If a generated EAD fails the EAD v3 schema.
    """
    return [
        analysis_app(analysis)
        for _, analysis in sorted(catalog.analyses.items())
        if analysis.empaia is not None
    ]


def analysis_app(analysis: Analysis) -> AnalysisApp:
    """Build the app of one analysis with `expose.empaia`.

    Args:
        analysis: The analysis.

    Returns:
        The app, its EAD schema-validated.

    Raises:
        ValueError: If the analysis has no `expose.empaia`.
        jsonschema.ValidationError: If the EAD fails the EAD v3 schema.
    """
    exposure = analysis.empaia
    if exposure is None:
        raise ValueError(f"{analysis.id} is not offered in EMPAIA")
    ead = generate_ead(analysis)
    configuration = {}
    if VIEWER_CONFIG_KEY in ead.get("configuration", {}).get("global", {}):
        settings = analysis.settings()
        viewer = exposure.viewer.with_input_defaults(
            {name: settings[name] for name in exposure.parameters}
        )
        configuration[VIEWER_CONFIG_KEY] = viewer.for_io(ead["io"])
    return AnalysisApp(
        analysis.id, exposure.modes, ead, configuration, exposure.triggers
    )


def generate_ead(analysis: Analysis) -> dict[str, Any]:
    """Merge the converters' and publishers' fragments into the analysis's EAD.

    Args:
        analysis: An analysis with `expose.empaia`.

    Returns:
        The schema-validated EAD.

    Raises:
        ValueError: If the analysis has no `expose.empaia`.
        jsonschema.ValidationError: If the EAD fails the EAD v3 schema.
    """
    exposure = analysis.empaia
    if exposure is None:
        raise ValueError(f"{analysis.id} is not offered in EMPAIA")
    mode_inputs = {
        mode: exposure.mode_inputs(mode, analysis.inputs) for mode in exposure.modes
    }
    used = {name for names in mode_inputs.values() for name in names}
    io: dict[str, Any] = {
        name: converter.io_fragment(EmpaiaInput(name, exposure.slide))
        for name, converter in exposure.inputs.items()
        if name in used
    }
    io |= {
        name: publisher.io_fragment(EmpaiaOutput(name, exposure.slide))
        for name, publisher in exposure.outputs.items()
    }
    parameters = exposure.parameters
    if (offer := exposure.parameter_converter) is not None:
        schema = analysis.settings_schema()["properties"]
        io |= {
            name: offer.io_fragment(EmpaiaInput(name, setting=schema[name]))
            for name in exposure.parameters
        }
    ead: dict[str, Any] = {
        "$schema": EAD_SCHEMA_URL,
        "name": exposure.name,
        "name_short": exposure.name_short,
        "namespace": f"{NAMESPACE_PREFIX}{_identifier(analysis.id)}.v3.0",
        "description": " ".join(
            [exposure.description, *(m.description for m in exposure.modes)]
        ),
        # The RationAI direct-access route serves apps allowed raw slide access.
        "permissions": {
            "data_transmission_to_external_service_provider": True,
            "wsi_raw_file_access": True,
        },
        "io": io,
        "modes": {
            mode.value: {
                "inputs": [
                    *names,
                    *(parameters if mode is AppMode.STANDALONE else ()),
                ],
                "outputs": list(exposure.outputs),
            }
            for mode, names in mode_inputs.items()
        },
    }
    # The viewer's configuration also carries the parameters' defaults.
    if exposure.viewer.inputs or exposure.viewer.outputs or parameters:
        ead["configuration"] = {
            "global": {
                VIEWER_CONFIG_KEY: {
                    "type": "string",
                    "description": "Configuration for the viewer",
                }
            }
        }
    _EAD_SCHEMA.validate(ead)
    return ead


def stable_id(kind: str, namespace: str) -> str:
    """Derive a UUID from the namespace, shaped as version 4 as the marketplace requires.

    Args:
        kind: What the ID names, e.g. `empaia-app`; keeps one namespace's IDs apart.
        namespace: The EAD namespace.

    Returns:
        The UUID as a string.
    """
    return str(UUID(bytes=uuid5(NAMESPACE_URL, f"{kind}:{namespace}").bytes, version=4))


def _tags(names: Mapping[str, dict[str, str]], ids: Iterable[str]) -> list[Any]:
    """Build a portal app's tag list, without `ANY`, which says nothing there.

    Args:
        names: The tag group's names per language, by tag ID.
        ids: The tag IDs, repeats allowed.

    Returns:
        One marketplace tag per distinct ID, in first-seen order.
    """
    return [
        {
            "name": tag,
            "tag_translations": [
                {"lang": lang, "text": text} for lang, text in names[tag].items()
            ],
        }
        for tag in dict.fromkeys(ids)
        if tag != ANY
    ]


def _identifier(value: str) -> str:
    """Replace each non-alphanumeric character with `_` for use in a namespace.

    Args:
        value: An analysis ID.

    Returns:
        The identifier.
    """
    return "".join(c if c.isalnum() else "_" for c in value)
