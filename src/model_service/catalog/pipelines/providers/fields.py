"""The keys of `providers:` and `providers.empaia` a catalog file may write."""

from __future__ import annotations

from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from model_service.providers.empaia.ead.vocabulary import Trigger, ViewerConfig


# EAD names allow letters, digits, spaces, `-` and `_`.
_TEXT = r"^[a-zA-Z0-9_\s\-]+$"


class _Strict(BaseModel):
    """A block of `providers:` keys: frozen, and unknown keys refused."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class StandaloneFields(_Strict):
    """Standalone mode: a user starts the app on a slide."""

    parameters: tuple[str, ...] = Field(
        (),
        description="Per-run settings users fill in per job; the rest keep the file's values.",
    )


class PreprocessingFields(_Strict):
    """Preprocessing mode: EMPAIA runs the app on new slides."""

    triggers: tuple[Trigger, ...] = Field(
        (), description="The tissue and stain tags of the slides EMPAIA preprocesses."
    )


class ModesFields(_Strict):
    """The job modes the app offers; a mode is on when its key is present."""

    standalone: StandaloneFields | None = Field(None, description="Standalone mode.")
    preprocessing: PreprocessingFields | None = Field(
        None, description="Preprocessing mode."
    )

    @model_validator(mode="before")
    @classmethod
    def _present_keys_turn_modes_on(cls, data: object) -> object:
        """Read a mode key without a value as the mode with no settings.

        Args:
            data: The `modes:` value as written.

        Returns:
            The value, `None` modes replaced by empty mappings.
        """
        if isinstance(data, dict):
            return {k: {} if v is None else v for k, v in data.items()}
        return data

    @model_validator(mode="after")
    def _at_least_one(self) -> Self:
        """Refuse an app that offers no mode."""
        if self.standalone is None and self.preprocessing is None:
            raise ValueError("modes needs standalone, preprocessing or both")
        return self


class EmpaiaFields(_Strict):
    """Every key of `providers.empaia`."""

    name: Annotated[str, Field(min_length=3, max_length=30, pattern=_TEXT)] = Field(
        description="The app's name in EMPAIA (3 to 30 letters, digits, spaces, `-`, `_`)."
    )
    name_short: Annotated[str, Field(min_length=3, max_length=10, pattern=_TEXT)] = (
        Field(description="The app's short name (3 to 10 characters).")
    )
    description: Annotated[str, Field(min_length=3, max_length=900)] = Field(
        description="What the app finds; each mode adds a sentence."
    )
    modes: ModesFields = Field(description="`standalone:`, `preprocessing:` or both.")
    inputs: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="EAD type per input, `{type: <type>, …options}`; slide is wsi "
        "and region_of_interest rectangle unless given.",
    )
    outputs: dict[str, dict[str, Any]] = Field(
        min_length=1,
        description="EAD type per published sink output, `{type: <type>, …options}`.",
    )
    viewer: ViewerConfig = Field(
        default_factory=ViewerConfig,
        description="The AigoPath case browser's display settings, by io key.",
    )


class ProvidersFields(_Strict):
    """Every key of `providers:`."""

    empaia: Any = Field(None, description="The EMPAIA app the pipeline becomes.")
