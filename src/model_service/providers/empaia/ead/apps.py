"""EMPAIA apps for the deployed models, one per builder, offering every listed mode.

A model file opts in with an `empaia` block (see `listing`). Every builder that
accepts the model's contract, is listed in `builder_io.BUILDER_IO`, and whose
outputs all have a publisher that takes the model then becomes one app with the
listed modes. In standalone mode it takes all of the builder's EMPAIA inputs
(e.g. a region of interest); in preprocessing mode only the required ones,
because EMPAIA starts it with a slide alone. The app's image is
`http::<target>/builders/<builder>/models/<model>`, which JES hands to
model-service. Builder settings the model file lists as parameters become
standalone inputs the user fills in per job; their type and description come
from the builder's configuration.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, UUID, uuid5

from jsonschema import Draft7Validator

from model_service.builders.registry import BUILDERS
from model_service.providers.empaia.ead.builder_io import BUILDER_IO
from model_service.providers.empaia.ead.listing import (
    PARAMETER_TYPES,
    VIEWER_CONFIG_KEY,
    AppMode,
    ViewerConfig,
)
from model_service.providers.empaia.tags import ANY, STAINS, TISSUES


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from model_service.builders.declarations import BuilderSpec
    from model_service.contracts.model_declarations import ModelBinding
    from model_service.providers.empaia.ead.builder_io import BuilderIo
    from model_service.providers.empaia.ead.listing import EmpaiaListing, Trigger
    from model_service.providers.empaia.publishing.publishers import PublisherRegistry


EAD_SCHEMA_URL = "https://gitlab.com/empaia/integration/definitions/-/raw/main/ead/ead-schema.v3.json"
_EAD_SCHEMA = Draft7Validator(
    json.loads(Path(__file__).with_name("ead-v3.schema.json").read_text())
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AppDefinition:
    """One EMPAIA app: a builder with a model.

    Attributes:
        builder_id: The builder the app runs.
        model_id: The model the builder uses.
        modes: The job modes the app supports.
        ead: The app's EAD.
        global_configuration: Values for the EAD's `configuration.global`,
            set when the app is registered.
        triggers: The slides EMPAIA preprocesses with the app.
        parameters: The builder settings users set per standalone job.
    """

    builder_id: str
    model_id: str
    modes: tuple[AppMode, ...]
    ead: dict[str, Any]
    # Values for the EAD's `configuration.global`, set when the app is registered.
    global_configuration: dict[str, str] = field(default_factory=dict)
    triggers: tuple[Trigger, ...] = ()
    parameters: tuple[str, ...] = ()

    def image(self, target: str) -> str:
        """Build the app's image reference, which JES hands to model-service.

        Args:
            target: model-service's name in JES's HTTP targets.

        Returns:
            `http::<target>/builders/<builder>/models/<model>`.
        """
        return f"http::{target}/builders/{self.builder_id}/models/{self.model_id}"

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

    @property
    def app_id(self) -> str:
        """The ID EMPAIA jobs name the app by."""
        return stable_id("empaia-app", self.ead["namespace"])

    @property
    def portal_app_id(self) -> str:
        """The ID the marketplace lists the app by, and triggers name it by."""
        return stable_id("empaia-portal-app", self.ead["namespace"])


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


def stable_id(kind: str, namespace: str) -> str:
    """Derive a UUID from the namespace, shaped as version 4 as the marketplace requires.

    Args:
        kind: What the ID names, e.g. `empaia-app`; keeps one namespace's IDs apart.
        namespace: The EAD namespace.

    Returns:
        The UUID as a string.
    """
    return str(UUID(bytes=uuid5(NAMESPACE_URL, f"{kind}:{namespace}").bytes, version=4))


def app_definitions(
    models: Iterable[ModelBinding],
    listings: Mapping[str, EmpaiaListing],
    publishers: PublisherRegistry,
    builders: Iterable[BuilderSpec] = BUILDERS.values(),
    io: Mapping[str, BuilderIo] = BUILDER_IO,
) -> list[AppDefinition]:
    """Build every app the listed models offer.

    A model without a listing offers none. A builder becomes an app of a model
    when it accepts the model's contract, has an `io` entry, and every one of its
    outputs has a publisher that accepts the model; otherwise it is skipped and
    logged.

    Args:
        models: The deployed models.
        listings: The EMPAIA listings, by model ID.
        publishers: The publishers outputs are checked against.
        builders: The builders to consider; all registered ones by default.
        io: The EAD io entries, by builder ID.

    Returns:
        One app per offered builder and model.

    Raises:
        ValueError: If an io entry references an input the app does not take.
        jsonschema.ValidationError: If a generated EAD fails the EAD v3 schema.
    """
    builders = list(builders)
    apps = []
    for model in models:
        listing = listings.get(model.id)
        if listing is None:
            continue
        for spec in builders:
            builder_io = io.get(spec.id)
            if spec.accepts is not model.contract or builder_io is None:
                continue
            if unpublishable := [
                name
                for name, output in spec.outputs.items()
                if (
                    publisher := publishers.resolve(
                        builder_io.output_type(name), output
                    )
                )
                is None
                or not publisher.accepts(model)
            ]:
                logger.info(
                    "%s with the %s builder is not offered in EMPAIA: "
                    "nothing publishes %s",
                    model.id,
                    spec.id,
                    ", ".join(unpublishable),
                )
                continue
            ead = generate_ead(spec, builder_io, model.id, listing)
            configuration = {}
            viewer = listing.viewer.get(spec.id)
            if parameters := listing.parameters.get(spec.id, ()):
                defaults = spec.with_defaults()
                viewer = (viewer or ViewerConfig()).with_input_defaults(
                    {name: defaults[name] for name in parameters}
                )
            if viewer is not None:
                configuration[VIEWER_CONFIG_KEY] = viewer.for_io(ead["io"])
            apps.append(
                AppDefinition(
                    spec.id,
                    model.id,
                    listing.modes,
                    ead,
                    configuration,
                    listing.triggers,
                    listing.parameters.get(spec.id, ()),
                )
            )
    return apps


def generate_ead(
    spec: BuilderSpec,
    builder_io: BuilderIo,
    model_id: str,
    listing: EmpaiaListing,
) -> dict[str, Any]:
    """Build and schema-validate the EAD of one app, with every listed mode.

    Standalone mode takes every input both the builder and `io` have, and the
    listed parameters; preprocessing mode only the required inputs, as EMPAIA
    starts it with a slide.

    Args:
        spec: The builder.
        builder_io: The builder's EAD io entries.
        model_id: The model, part of the namespace.
        listing: How the model appears in EMPAIA.

    Returns:
        The EAD.

    Raises:
        ValueError: If an io entry references an input the app does not take.
        jsonschema.ValidationError: If the EAD fails the EAD v3 schema.
    """
    offered = [name for name in spec.inputs if name in builder_io.inputs]
    mode_inputs = {
        # EMPAIA starts a preprocessing job with the slide alone.
        mode: [
            name
            for name in offered
            if spec.inputs[name].required or mode is AppMode.STANDALONE
        ]
        for mode in listing.modes
    }
    inputs = {
        name: builder_io.inputs[name]
        for name in offered
        if any(name in names for names in mode_inputs.values())
    }
    outputs = {name: builder_io.outputs[name] for name in spec.outputs}
    io = {}
    for name, mapping in {**inputs, **outputs}.items():
        entry = dict(mapping or {})
        if "reference" in entry:
            target = entry["reference"]
            if target not in inputs:
                raise ValueError(f"{name!r} references unknown input {target!r}")
            entry["reference"] = f"io.{target}"
        io[name] = entry
    parameters = listing.parameters.get(spec.id, ())
    settings = spec.configuration.model_fields
    for name in parameters:
        io[name] = {
            "type": PARAMETER_TYPES[settings[name].annotation],
            "description": settings[name].description,
        }
    ead = {
        "$schema": EAD_SCHEMA_URL,
        "name": f"{listing.name} {spec.id}",
        "name_short": listing.name_short,
        "namespace": (
            f"org.empaia.rationai.{_identifier(spec.id)}_{_identifier(model_id)}.v3.0"
        ),
        "description": " ".join(
            [listing.description, *(m.description for m in listing.modes)]
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
                "outputs": list(outputs),
            }
            for mode, names in mode_inputs.items()
        },
    }
    # The viewer's configuration also carries the parameters' defaults.
    if spec.id in listing.viewer or parameters:
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


def _identifier(value: str) -> str:
    """Replace each non-alphanumeric character with `_` for use in a namespace.

    Args:
        value: A builder or model ID.

    Returns:
        The identifier.
    """
    return "".join(c if c.isalnum() else "_" for c in value)
