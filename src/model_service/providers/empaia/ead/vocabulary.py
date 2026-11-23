"""EMPAIA vocabulary an analysis's `expose.empaia` uses: modes, triggers, viewer settings.

Light on purpose: the catalog loader validates `expose.empaia` with these
models, and the EMPAIA provider builds EADs from them.
"""

from __future__ import annotations

import json
import math
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from model_service.providers.empaia.tags import STAINS, TISSUES


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


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


def _plain(value: JsonValue) -> bool:
    """Whether a JSON value is a finite number, a string or a boolean.

    Args:
        value: The value.

    Returns:
        True for a plain scalar.
    """
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, (bool, int, str))


class ViewerShader(_ViewerEntry):
    """An xOpat shader for an output.

    Attributes:
        type: The shader, e.g. `heatmap` or `colormap`.
        params: The shader's parameters.
    """

    # An xOpat shader, e.g. `heatmap` or `colormap`, with its parameters.
    type: str = Field(min_length=1)
    params: dict[str, JsonValue] | None = Field(
        None,
        description=(
            'The shader\'s parameters, e.g. `{threshold: 65, color: "#ffff00"}`: '
            "numbers, strings, booleans or lists of them."
        ),
    )

    @field_validator("params")
    @classmethod
    def _flat(cls, params: dict[str, JsonValue] | None) -> dict[str, JsonValue] | None:
        """Refuse nested or non-finite shader parameters.

        xOpat shader parameters are a number, a colour string, a flag, or a
        list of those.

        Args:
            params: The parameters as written.

        Returns:
            The parameters.

        Raises:
            ValueError: If a value is a mapping, a nested list, null, NaN or
                infinity.
        """
        for name, value in (params or {}).items():
            items = value if isinstance(value, list) else [value]
            if not all(_plain(item) for item in items):
                raise ValueError(
                    f"shader parameter `{name}` must be a number, string, "
                    "boolean or a list of them"
                )
        return params


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
