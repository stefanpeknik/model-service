"""What a builder declares: the contract it accepts, its inputs, configuration and outputs.

A builder's declaration lives in its package's `spec.py` and imports only
this module and Pydantic, so callers can validate requests and list builders
without the slide-processing dependencies. The input and output declarations
are derived from the builder's `BuilderInputs` and `BuilderOutputs` models.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, ValidationError

from model_service.contracts.failures import IncompatibleModelError, InvalidInvocation
from model_service.contracts.model_declarations import InferenceContract
from model_service.contracts.result import (
    BuilderResult,
    InlineOutput,
    OutputKind,
    SharedArtifact,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.contracts.invocation import Invocation
    from model_service.contracts.json_values import JsonValue
    from model_service.contracts.model_declarations import ModelBinding


@dataclass(frozen=True, slots=True)
class Artifact:
    """Declares a `SharedArtifact` output field: its media type and default file name.

    Used as `Annotated[SharedArtifact, Artifact(...)]` in a `BuilderOutputs` model.
    """

    media_type: str | None = None
    # Output file name when the invocation names no output path.
    default_path: str | None = None


class BuilderOutputs(BaseModel):
    """A builder's outputs, one field per output.

    A field is a `SharedArtifact`, annotated with `Artifact`, or an
    `InlineOutput`. A builder returns its own subclass; the registry derives the
    builder's output declaration from the fields, and the application turns it
    into the generic `BuilderResult` that is sent and stored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    def result(self) -> BuilderResult:
        """Return the outputs as the generic result that is sent and stored.

        Returns:
            A `BuilderResult` with one entry per field.
        """
        return BuilderResult(outputs=dict(self))


@dataclass(frozen=True, slots=True)
class Input:
    """Declares a `BuilderInputs` field's media type.

    Used as `Annotated[Path, Input(...)]`, or `Annotated[Path | None, Input(...)]`
    for an optional input.
    """

    media_type: str | None = None


class BuilderInputs(BaseModel):
    """A builder's inputs, one field per input, as local files.

    A field is a `Path` (required) or `Path | None` defaulting to None
    (optional), annotated with `Input`. The registry derives the builder's input
    declaration from the fields; the application resolves each supplied input to
    a local file through the input access and hands the builder an instance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    def paths(self) -> tuple[Path, ...]:
        """Return the supplied inputs' paths.

        Returns:
            One path per input the invocation supplied, in field order.
        """
        return tuple(path for path in dict(self).values() if path is not None)


@dataclass(frozen=True, slots=True)
class InputSpec:
    required: bool = True
    media_type: str | None = None


@dataclass(frozen=True, slots=True)
class OutputSpec:
    kind: OutputKind
    media_type: str | None = None
    default_path: str | None = None


def input_specs(inputs: type[BuilderInputs]) -> dict[str, InputSpec]:
    """Derive an input declaration from a builder's inputs model.

    Args:
        inputs: The builder's `BuilderInputs` subclass.

    Returns:
        One `InputSpec` per field, in field order.

    Raises:
        TypeError: A field is neither a required `Path` nor an optional
            `Path | None` defaulting to None.
    """
    specs = {}
    for name, info in inputs.model_fields.items():
        marker = next((m for m in info.metadata if isinstance(m, Input)), Input())
        annotation = info.annotation
        optional = get_origin(annotation) in {Union, types.UnionType} and set(
            get_args(annotation)
        ) == {Path, type(None)}
        if annotation is Path and info.is_required():
            specs[name] = InputSpec(media_type=marker.media_type)
        elif optional and info.default is None:
            specs[name] = InputSpec(required=False, media_type=marker.media_type)
        else:
            raise TypeError(f"{inputs.__name__}.{name} is not a builder input")
    return specs


def output_specs(outputs: type[BuilderOutputs]) -> dict[str, OutputSpec]:
    """Derive an output declaration from a builder's outputs model.

    Args:
        outputs: The builder's `BuilderOutputs` subclass.

    Returns:
        One `OutputSpec` per field, in field order.

    Raises:
        TypeError: A field is neither a `SharedArtifact` nor an `InlineOutput`.
    """
    specs = {}
    for name, info in outputs.model_fields.items():
        if info.annotation is InlineOutput:
            specs[name] = OutputSpec(kind=OutputKind.INLINE)
        elif info.annotation is SharedArtifact:
            artifact = next(
                (m for m in info.metadata if isinstance(m, Artifact)), Artifact()
            )
            specs[name] = OutputSpec(
                kind=OutputKind.SHARED_ARTIFACT,
                media_type=artifact.media_type,
                default_path=artifact.default_path,
            )
        else:
            raise TypeError(f"{outputs.__name__}.{name} is not a builder output")
    return specs


@dataclass(frozen=True, slots=True)
class BuilderSpec:
    id: str
    accepts: InferenceContract
    # What the builder receives and returns; `inputs` and `outputs` derive from them.
    input_model: type[BuilderInputs]
    output_model: type[BuilderOutputs]
    configuration: type[BaseModel]
    # "module:Class" of the `Builder`, imported only in builder replicas.
    implementation: str
    inputs: Mapping[str, InputSpec] = field(init=False)
    outputs: Mapping[str, OutputSpec] = field(init=False)

    def __post_init__(self) -> None:
        """Derive the input and output declarations from their models."""
        object.__setattr__(self, "inputs", input_specs(self.input_model))
        object.__setattr__(self, "outputs", output_specs(self.output_model))

    @property
    def placeable_output(self) -> str | None:
        """The single artifact output an invocation's `output_path` can place.

        Returns:
            The name of the only output with a default path, or None when there
            are none or several.
        """
        names = [name for name, output in self.outputs.items() if output.default_path]
        return names[0] if len(names) == 1 else None

    def check_invocation(
        self, invocation: Invocation, *, now: datetime | None = None
    ) -> None:
        """Check an invocation against this declaration before running it.

        Args:
            invocation: Invocation to check.
            now: Time to compare the deadline against; the current UTC time if None.

        Raises:
            InvalidInvocation: The deadline lacks a timezone or has passed, inputs
                are unknown or missing, the configuration is invalid, or
                `output_path` is set for a builder without exactly one artifact
                output.
        """
        deadline = invocation.deadline_at
        if deadline.tzinfo is None or deadline.utcoffset() is None:
            raise InvalidInvocation("deadline_at must include a timezone")
        if deadline <= (now or datetime.now(UTC)):
            raise InvalidInvocation("deadline_at must be in the future")
        if unknown := invocation.inputs.keys() - self.inputs.keys():
            raise InvalidInvocation(f"unknown inputs: {', '.join(sorted(unknown))}")
        if missing := {
            name
            for name, item in self.inputs.items()
            if item.required and name not in invocation.inputs
        }:
            raise InvalidInvocation(f"missing inputs: {', '.join(sorted(missing))}")
        if invocation.resolved_configuration is not None:
            self.resolve_configuration(invocation.resolved_configuration)
        if invocation.output_path is not None and self.placeable_output is None:
            raise InvalidInvocation(f"{self.id} has no artifact output to place")

    def check_model(self, model: ModelBinding) -> None:
        """Check that a model's inference contract is the one this builder accepts.

        Args:
            model: Model binding resolved for the attempt.

        Raises:
            IncompatibleModelError: The model's contract differs from `accepts`.
        """
        if model.contract != self.accepts:
            raise IncompatibleModelError(
                f"model {model.id!r} ({model.contract}) is incompatible with "
                f"builder {self.id!r} ({self.accepts})"
            )

    def resolve_configuration(
        self, overrides: Mapping[str, JsonValue] | None = None
    ) -> BaseModel:
        """Apply caller or provider overrides to the builder's current defaults.

        Args:
            overrides: Configuration fields to override; none uses the defaults.

        Returns:
            The validated configuration.

        Raises:
            InvalidInvocation: The overrides are invalid; the message names the
                fields only, since values may come from callers.
        """
        try:
            return self.configuration.model_validate(dict(overrides or {}))
        except ValidationError as error:
            # Field names only: values may come from callers.
            fields = sorted({".".join(map(str, e["loc"])) for e in error.errors()})
            raise InvalidInvocation(
                f"{self.id} configuration is invalid: {', '.join(fields)}"
            ) from error
