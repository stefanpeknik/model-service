"""EMPAIA vocabulary a pipeline's `providers.empaia` uses: modes, triggers, viewer settings.

Light on purpose: the catalog loader validates `providers.empaia` with these
models, and the EMPAIA provider builds EADs from them.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from model_service.providers.empaia.tags import STAINS, TISSUES


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
    """Base of the case browser's settings for one io key; unknown fields are refused."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str | None = Field(
        None, min_length=1, description="Name shown instead of the io key."
    )
    description: str | None = Field(None, description="Help text; replaces the EAD's.")


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


class ViewerShader(BaseModel):
    """An xOpat shader for an output.

    Attributes:
        type: The shader, e.g. `heatmap` or `colormap`.
        params: The shader's parameters.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

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
    """How the case browser shows an app input; each setting fits some input types only.

    A parameter's default and bounds come from its setting, so they are not
    written here. Which settings an input type takes is the case browser's
    schema (`aigopath-viewer.schema.json`), checked when the app is built.

    Attributes:
        prompt: Parameters: show the field (true) or send the default unseen.
        step: Numbers: the stepper's increment.
        unit: Numbers: the unit shown beside the field.
        multiline: Strings: a text area instead of one line.
        max_length: Strings: the longest accepted text.
        keep_visible: Drawn shapes: keep them on the slide after submitting.
        render: Slides: `background` or `none`.
    """

    prompt: bool | None = None
    step: float | None = Field(None, gt=0)
    unit: str | None = Field(None, min_length=1)
    multiline: bool | None = None
    max_length: int | None = Field(None, ge=1, le=200)
    keep_visible: bool | None = None
    render: Literal["background", "none"] | None = None


class ViewerOutput(_ViewerEntry):
    """How the case browser shows an app output.

    Attributes:
        render: How the output is drawn: `shader`, `background`, `metric` or
            `none`, as its type allows.
        visible: Whether an overlay is on when first shown.
        shader: The overlay's xOpat shader.
    """

    render: Literal["shader", "background", "metric", "none"] | None = None
    visible: bool | None = None
    shader: ViewerShader | None = None


class ViewerConfig(BaseModel):
    """How the AigoPath case browser shows an app's inputs and outputs, by io key.

    Outputs are layered in the order written, the first at the bottom.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    inputs: dict[str, ViewerInput] = Field(default_factory=dict)
    outputs: dict[str, ViewerOutput] = Field(default_factory=dict)


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
