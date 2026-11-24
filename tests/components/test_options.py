"""Per-run options: only fields marked `PerRun` take a caller's values."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from components import per_run_fields, with_overrides
from tests.support.components.steps.mean_intensity import MeanIntensityOptions


FILE_OPTIONS = MeanIntensityOptions(channels=2, scale=1.0)


def test_only_per_run_fields_are_settings() -> None:
    assert per_run_fields(MeanIntensityOptions) == {"scale"}


def test_a_per_run_override_replaces_only_its_own_value() -> None:
    options = with_overrides(FILE_OPTIONS, {"scale": 2.5})

    assert (options.channels, options.scale) == (2, 2.5)


def test_an_override_of_a_fixed_option_is_refused() -> None:
    with pytest.raises(ValueError, match=r"\['channels'\] are fixed.*\['scale'\]"):
        with_overrides(FILE_OPTIONS, {"channels": 3})


def test_an_override_is_validated_like_the_file() -> None:
    with pytest.raises(ValidationError):
        with_overrides(FILE_OPTIONS, {"scale": -1})
