"""A component's options and the per-run marker.

Options come from the catalog file. A field marked `PerRun[...]` may also be
overridden by a caller for one run; those fields are the analysis's API
settings and EMPAIA parameters. Every other field is fixed by the file.
"""

from __future__ import annotations

from typing import Annotated, TypeVar

from pydantic import BaseModel, ConfigDict


T = TypeVar("T")


class PerRunMarker:
    """Field metadata that lets a caller override the option per run."""

    def __repr__(self) -> str:
        """Name the marker in pydantic's field reprs."""
        return "PerRun"


PER_RUN = PerRunMarker()

PerRun = Annotated[T, PER_RUN]
"""Marks an option a caller may override per run: `stride: PerRun[float] = 0.5`."""


class Options(BaseModel):
    """Base of every component's options: unknown keys, NaN and infinity are errors."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def per_run_fields(options: type[BaseModel]) -> frozenset[str]:
    """The option names a caller may override per run.

    Args:
        options: A component's `Options` class.

    Returns:
        The names of the fields marked `PerRun`.
    """
    return frozenset(
        name
        for name, info in options.model_fields.items()
        if any(isinstance(meta, PerRunMarker) for meta in info.metadata)
    )


def with_overrides[O: BaseModel](options: O, overrides: dict[str, object]) -> O:
    """Validate a caller's per-run values over options from the catalog file.

    Args:
        options: The options the catalog file gives.
        overrides: The caller's values, keyed by option name.

    Returns:
        New options with the overrides applied and validated.

    Raises:
        ValueError: If an override names an option that is not `PerRun`.
        pydantic.ValidationError: If an override has an invalid value.
    """
    allowed = per_run_fields(type(options))
    fixed = sorted(set(overrides) - allowed)
    if fixed:
        raise ValueError(
            f"options {fixed} are fixed by the catalog file; "
            f"per-run options: {sorted(allowed) or 'none'}"
        )
    return type(options).model_validate(options.model_dump() | overrides)
