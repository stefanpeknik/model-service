"""EMPAIA apps for the deployed models, one per builder and mode.

A model file opts in with an `empaia` block: how the model is presented and in
which modes it runs. Every builder that accepts the model's contract and has an
EMPAIA type for each output then becomes one app per mode. A standalone app
takes all of the builder's EMPAIA inputs (e.g. a region of interest); a
preprocessing app takes only the required ones, because EMPAIA starts it with a
slide alone. The app's image is `http::<target>/builders/<builder>/models/<model>`,
which JES hands to model-service.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from jsonschema import Draft7Validator
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from model_service.builders.contracts import SpatialScoresModel
from model_service.builders.registry import BUILDERS


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from model_service.builders.contracts import ModelBinding
    from model_service.builders.registry import BuilderSpec


EAD_SCHEMA_URL = "https://gitlab.com/empaia/integration/definitions/-/raw/main/ead/ead-schema.v3.json"
_EAD_SCHEMA = Draft7Validator(
    json.loads(Path(__file__).with_name("ead-v3.schema.json").read_text())
)
# EAD names allow letters, digits, spaces, `-` and `_`.
_TEXT = r"^[a-zA-Z0-9_\s\-]+$"


class AppMode(StrEnum):
    STANDALONE = "standalone"
    PREPROCESSING = "preprocessing"


_MODE_DESCRIPTION = {
    AppMode.STANDALONE: (
        "Runs on the region of interest you draw, or on the whole slide without one."
    ),
    AppMode.PREPROCESSING: "Runs on the whole slide when the slide is preprocessed.",
}


# The app's global configuration key the AigoPath case browser reads its
# display settings from: a JSON string shaped as `ViewerConfig`.
VIEWER_CONFIG_KEY = "aigopath_viewer"


class _ViewerEntry(BaseModel):
    # The case browser passes fields it doesn't know on to xOpat.
    model_config = ConfigDict(frozen=True, extra="allow")


class ViewerShader(_ViewerEntry):
    # An xOpat shader, e.g. `heatmap` or `colormap`, with its parameters.
    type: str
    params: dict[str, JsonValue] | None = None


class ViewerInput(_ViewerEntry):
    target: str | None = None
    lossless: bool | None = None
    label: str | None = None


class ViewerOutput(_ViewerEntry):
    render: Literal["shader", "background", "metric", "none"] | None = None
    label: str | None = None
    visible: bool | None = None
    shader: ViewerShader | None = None


class ViewerConfig(BaseModel):
    """How the AigoPath case browser shows an app's inputs and outputs, by io key."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    inputs: dict[str, ViewerInput] = Field(default_factory=dict)
    outputs: dict[str, ViewerOutput] = Field(default_factory=dict)

    def for_io(self, keys: Iterable[str]) -> str:
        """The configuration value for an app with these io keys."""
        keys = set(keys)
        return json.dumps(
            {
                "inputs": {
                    k: v.model_dump(exclude_none=True)
                    for k, v in self.inputs.items()
                    if k in keys
                },
                "outputs": {
                    k: v.model_dump(exclude_none=True)
                    for k, v in self.outputs.items()
                    if k in keys
                },
            }
        )


class EmpaiaListing(BaseModel):
    """How a model appears in EMPAIA; the `empaia` block of its model file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The app is named `<name> <builder>`, e.g. "Prostate cancer heatmap".
    name: Annotated[str, Field(min_length=3, max_length=22, pattern=_TEXT)]
    name_short: Annotated[str, Field(min_length=3, max_length=10, pattern=_TEXT)]
    # What the model finds, for someone choosing an app; the mode adds a sentence.
    description: Annotated[str, Field(min_length=3, max_length=900)]
    modes: Annotated[tuple[AppMode, ...], Field(min_length=1)]
    # Entries for io keys an app lacks are left out of that app's value.
    viewer: ViewerConfig | None = None


@dataclass(frozen=True, slots=True)
class AppDefinition:
    builder_id: str
    model_id: str
    mode: AppMode
    ead: dict[str, Any]
    # Values for the EAD's `configuration.global`, set when the app is registered.
    global_configuration: dict[str, str] = field(default_factory=dict)

    def image(self, target: str) -> str:
        return f"http::{target}/builders/{self.builder_id}/models/{self.model_id}"

    def portal_app(self, target: str, organization_id: str) -> dict[str, Any]:
        """The marketplace mock's portal app.

        Its IDs derive from the namespace, so registering it again replaces it.
        """
        namespace = self.ead["namespace"]
        version = namespace[namespace.rfind(".") - 2 :]
        now = int(time.time())
        common = {
            "portal_app_id": stable_id("empaia-portal-app", namespace),
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
                        "tissues": [],
                        "stains": [],
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

    @property
    def app_id(self) -> str:
        """The ID EMPAIA jobs name the app by."""
        return stable_id("empaia-app", self.ead["namespace"])


def stable_id(kind: str, namespace: str) -> str:
    """A UUID derived from the namespace, shaped as version 4 as the marketplace requires."""
    return str(UUID(bytes=uuid5(NAMESPACE_URL, f"{kind}:{namespace}").bytes, version=4))


def app_definitions(
    models: Iterable[ModelBinding],
    listings: Mapping[str, EmpaiaListing],
    builders: Iterable[BuilderSpec] = BUILDERS.values(),
) -> list[AppDefinition]:
    """Every app the listed models offer."""
    builders = list(builders)
    apps = []
    for model in models:
        listing = listings.get(model.id)
        if listing is None:
            continue
        # The SDK publishes single-channel pixelmaps only.
        if isinstance(model, SpatialScoresModel) and model.n_channels != 1:
            continue
        for spec in builders:
            if spec.accepts is not model.contract or not _publishable(spec):
                continue
            for mode in listing.modes:
                ead = generate_ead(spec, model.id, listing, mode)
                configuration = {}
                if listing.viewer is not None:
                    configuration[VIEWER_CONFIG_KEY] = listing.viewer.for_io(ead["io"])
                apps.append(AppDefinition(spec.id, model.id, mode, ead, configuration))
    return apps


def generate_ead(
    spec: BuilderSpec, model_id: str, listing: EmpaiaListing, mode: AppMode
) -> dict[str, Any]:
    """Build and schema-validate the EAD of one app."""
    inputs = {
        name: item.empaia
        for name, item in spec.inputs.items()
        if item.empaia and (item.required or mode is AppMode.STANDALONE)
    }
    outputs = {name: item.empaia for name, item in spec.outputs.items()}
    io = {}
    for name, mapping in {**inputs, **outputs}.items():
        entry = dict(mapping or {})
        if "reference" in entry:
            target = entry["reference"]
            if target not in inputs:
                raise ValueError(f"{name!r} references unknown input {target!r}")
            entry["reference"] = f"io.{target}"
        io[name] = entry
    ead = {
        "$schema": EAD_SCHEMA_URL,
        "name": f"{listing.name} {spec.id}",
        "name_short": listing.name_short,
        "namespace": (
            f"org.empaia.rationai.{_identifier(spec.id)}_{_identifier(model_id)}"
            f"_{mode.value}.v3.0"
        ),
        "description": f"{listing.description} {_MODE_DESCRIPTION[mode]}",
        # The RationAI direct-access route serves apps allowed raw slide access.
        "permissions": {
            "data_transmission_to_external_service_provider": True,
            "wsi_raw_file_access": True,
        },
        "io": io,
        "modes": {mode.value: {"inputs": list(inputs), "outputs": list(outputs)}},
    }
    if listing.viewer is not None:
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


def _publishable(spec: BuilderSpec) -> bool:
    """EMPAIA has a type for every output, e.g. not for HDF5 embeddings."""
    return all(output.empaia for output in spec.outputs.values())


def _identifier(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in value)
