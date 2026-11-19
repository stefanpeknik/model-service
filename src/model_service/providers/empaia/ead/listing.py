"""The `empaia` block of a model file: how the model appears in EMPAIA.

It names the model's apps, the job modes they offer, the case browser's display
settings per builder, the slides EMPAIA preprocesses with them, and the builder
settings users set per standalone job. `parse_listings` validates the blocks;
`apps` turns them into EADs.
"""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from model_service.builders.registry import BUILDERS
from model_service.providers.empaia.tags import STAINS, TISSUES


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


logger = logging.getLogger(__name__)

# EAD names allow letters, digits, spaces, `-` and `_`.
_TEXT = r"^[a-zA-Z0-9_\s\-]+$"
# The EMPAIA primitive type of each builder setting type a parameter may have.
PARAMETER_TYPES: dict[object, str] = {
    bool: "bool",
    int: "integer",
    float: "float",
    str: "string",
}


class AppMode(StrEnum):
    """The EMPAIA job mode an app supports."""

    STANDALONE = "standalone"
    PREPROCESSING = "preprocessing"

    @property
    def description(self) -> str:
        """The sentence an app's description gets for this mode."""
        if self is AppMode.STANDALONE:
            return "Runs on the region of interest you draw."
        return "Runs on the whole slide when the slide is preprocessed."


# The app's global configuration key the AigoPath case browser reads its
# display settings from: a JSON string shaped as `ViewerConfig`.
VIEWER_CONFIG_KEY = "aigopath_viewer"


class _ViewerEntry(BaseModel):
    """Base of the case browser's viewer settings; unknown fields pass through to xOpat."""

    # The case browser passes fields it doesn't know on to xOpat.
    model_config = ConfigDict(frozen=True, extra="allow")


class ViewerShader(_ViewerEntry):
    """An xOpat shader for an output.

    Attributes:
        type: The shader, e.g. `heatmap` or `colormap`.
        params: The shader's parameters.
    """

    # An xOpat shader, e.g. `heatmap` or `colormap`, with its parameters.
    type: str
    params: dict[str, JsonValue] | None = None


class ViewerInput(_ViewerEntry):
    """How the viewer shows an app input; a parameter's entry also has its `default`.

    Attributes:
        target: The input the entry applies to.
        lossless: Whether the viewer loads it without lossy compression.
        label: Label shown in the viewer.
    """

    target: str | None = None
    lossless: bool | None = None
    label: str | None = None


class ViewerOutput(_ViewerEntry):
    """How the viewer shows an app output.

    Attributes:
        render: How the output is drawn.
        label: Label shown in the viewer.
        visible: Whether it is shown by default.
        shader: The shader used when `render` is `shader`.
    """

    render: Literal["shader", "background", "metric", "none"] | None = None
    label: str | None = None
    visible: bool | None = None
    shader: ViewerShader | None = None


class ViewerConfig(BaseModel):
    """How the AigoPath case browser shows an app's inputs and outputs, by io key."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    inputs: dict[str, ViewerInput] = Field(default_factory=dict)
    outputs: dict[str, ViewerOutput] = Field(default_factory=dict)

    def with_input_defaults(self, defaults: Mapping[str, JsonValue]) -> ViewerConfig:
        """Add a `default` to the entries of inputs the user fills in, e.g. parameters.

        The case browser may pre-fill its job form from them; the job still
        sends every value explicitly.

        Args:
            defaults: The default value of each input, by io key.

        Returns:
            A copy with the defaults set; other entries are unchanged.
        """
        inputs = dict(self.inputs)
        for key, value in defaults.items():
            entry = inputs.get(key, ViewerInput()).model_dump(exclude_none=True)
            inputs[key] = ViewerInput.model_validate({**entry, "default": value})
        return self.model_copy(update={"inputs": inputs})

    def for_io(self, keys: Iterable[str]) -> str:
        """Serialize the configuration value for an app with these io keys.

        Entries for io keys the app lacks are left out.

        Args:
            keys: The app's EAD io keys.

        Returns:
            A JSON string with `inputs` and `outputs`, unset fields omitted.
        """
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


class Trigger(BaseModel):
    """Slides whose preprocessing runs the model's apps: one tissue and stain pair.

    Attributes:
        tissue: An EMPAIA tissue tag, e.g. `PROSTATE`, or `ANY`.
        stain: An EMPAIA stain tag, e.g. `H_AND_E`, or `ANY`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tissue: str
    stain: str

    @field_validator("tissue")
    @classmethod
    def _known_tissue(cls, tissue: str) -> str:
        """Reject a tissue EMPAIA has no tag for; a slide never carries it.

        Args:
            tissue: The tissue tag.

        Returns:
            The tag, unchanged.

        Raises:
            ValueError: If it is not an EMPAIA tissue tag.
        """
        if tissue not in TISSUES:
            raise ValueError(f"{tissue!r} is not an EMPAIA tissue tag")
        return tissue

    @field_validator("stain")
    @classmethod
    def _known_stain(cls, stain: str) -> str:
        """Reject a stain EMPAIA has no tag for; a slide never carries it.

        Args:
            stain: The stain tag.

        Returns:
            The tag, unchanged.

        Raises:
            ValueError: If it is not an EMPAIA stain tag.
        """
        if stain not in STAINS:
            raise ValueError(f"{stain!r} is not an EMPAIA stain tag")
        return stain


class EmpaiaListing(BaseModel):
    """How a model appears in EMPAIA; the `empaia` block of its model file.

    Attributes:
        name: The model's part of each app name, `<name> <builder>`.
        name_short: The short app name.
        description: What the model finds; each mode adds a sentence.
        modes: The job modes its apps offer.
        viewer: The case browser's display settings, by builder ID.
        triggers: The slides EMPAIA preprocesses with its apps.
        parameters: By builder ID, the settings users set per standalone job;
            their type and description come from the builder.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The app is named `<name> <builder>`, e.g. "Prostate cancer heatmap".
    name: Annotated[str, Field(min_length=3, max_length=22, pattern=_TEXT)]
    name_short: Annotated[str, Field(min_length=3, max_length=10, pattern=_TEXT)]
    # What the model finds, for someone choosing an app; the mode adds a sentence.
    description: Annotated[str, Field(min_length=3, max_length=900)]
    modes: Annotated[tuple[AppMode, ...], Field(min_length=1)]
    # Viewer settings per builder ID, since each builder is its own app with its
    # own outputs; entries for io keys an app lacks are left out of its value.
    viewer: dict[str, ViewerConfig] = Field(default_factory=dict)
    # The slides EMPAIA preprocesses with the model's apps; registration keeps
    # the installation's preprocessing triggers in line with them.
    triggers: tuple[Trigger, ...] = ()
    # Per builder ID, the settings users set per standalone job; the rest run
    # at the builder's defaults.
    parameters: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _modes_take_triggers_and_parameters(self) -> Self:
        """Reject triggers or parameters for a mode the model does not run in.

        Returns:
            The listing, unchanged.

        Raises:
            ValueError: If triggers are listed without the preprocessing mode, or
                parameters without the standalone mode.
        """
        if self.triggers and AppMode.PREPROCESSING not in self.modes:
            raise ValueError("triggers need the preprocessing mode")
        if any(self.parameters.values()) and AppMode.STANDALONE not in self.modes:
            raise ValueError("parameters need the standalone mode")
        return self

    @field_validator("parameters")
    @classmethod
    def _settings_users_can_set(
        cls, parameters: dict[str, tuple[str, ...]]
    ) -> dict[str, tuple[str, ...]]:
        """Accept only described, single-valued settings of registered builders.

        Args:
            parameters: The parameter names, by builder ID.

        Returns:
            The parameters, unchanged.

        Raises:
            ValueError: If a builder is not registered, or a name is repeated, is
                not one of its settings, has no description, is not a bool,
                integer, float or string, or is also an input or output name.
        """
        for builder_id, names in parameters.items():
            spec = BUILDERS.get(builder_id)
            if spec is None:
                raise ValueError(f"parameters name unknown builder {builder_id!r}")
            if len(set(names)) != len(names):
                raise ValueError(f"{builder_id} parameters repeat a name")
            for name in names:
                setting = spec.configuration.model_fields.get(name)
                if setting is None:
                    raise ValueError(f"{builder_id} has no setting {name!r}")
                if setting.annotation not in PARAMETER_TYPES:
                    raise ValueError(
                        f"{builder_id} setting {name!r} is not a bool, integer, "
                        "float or string"
                    )
                if not setting.description:
                    raise ValueError(
                        f"{builder_id} setting {name!r} has no description to show"
                    )
                if name in spec.inputs or name in spec.outputs:
                    raise ValueError(f"{builder_id} parameter {name!r} is also io")
        return parameters

    @field_validator("viewer")
    @classmethod
    def _known_builders(
        cls, viewer: dict[str, ViewerConfig]
    ) -> dict[str, ViewerConfig]:
        """Reject viewer settings for a builder that is not registered.

        Args:
            viewer: The viewer settings, by builder ID.

        Returns:
            The settings, unchanged.

        Raises:
            ValueError: If a key is not a registered builder ID.
        """
        if unknown := sorted(viewer.keys() - BUILDERS.keys()):
            raise ValueError(f"viewer names unknown builders: {', '.join(unknown)}")
        return viewer


def parse_listings(blocks: Mapping[str, object]) -> dict[str, EmpaiaListing]:
    """Validate the `empaia` blocks of the model files into listings.

    An invalid block is logged and its model left out of EMPAIA, so a mistake in
    one listing does not stop the API or the other models' apps.

    Args:
        blocks: Each model's `empaia` block, by model ID.

    Returns:
        The valid listings, by model ID.
    """
    listings = {}
    for model_id, block in blocks.items():
        try:
            listings[model_id] = EmpaiaListing.model_validate(block)
        except ValidationError as error:
            logger.error(
                "model %r is not offered in EMPAIA: its listing is invalid: %s",
                model_id,
                error,
            )
    return listings
