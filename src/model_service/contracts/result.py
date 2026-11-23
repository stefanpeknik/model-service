"""A run's result as it is sent and stored: one output per declared name."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    Field,
    TypeAdapter,
)

from model_service.contracts.json_values import (
    JsonValue,
    NonEmptyString,
)


class OutputKind(StrEnum):
    """How an output is returned: inline or as a shared artifact."""

    INLINE = "inline"
    SHARED_ARTIFACT = "shared_artifact"


@dataclass(frozen=True, slots=True)
class InlineOutput:
    """An output returned in the result itself.

    Attributes:
        value: The JSON value.
        kind: Always `inline`.
    """

    value: JsonValue = None
    kind: Literal[OutputKind.INLINE] = OutputKind.INLINE


@dataclass(frozen=True, slots=True)
class SharedArtifact:
    """A completed output file; the result carries its path, not its bytes."""

    path: NonEmptyString
    media_type: Annotated[str, Field(strict=True)] | None = None
    kind: Literal[OutputKind.SHARED_ARTIFACT] = OutputKind.SHARED_ARTIFACT


type RunOutput = Annotated[InlineOutput | SharedArtifact, Field(discriminator="kind")]


@dataclass(frozen=True, slots=True)
class ServedModelRun:
    """What a served model ran during a run, as its `describe()` reported it.

    Attributes:
        digest: The served model's declaration digest.
        weights_revision: The weights it loaded, e.g. `<repo>@<commit>`.
        execution_providers: What the runtime actually runs on: ONNX Runtime's
            providers in priority order, or the torch device, so a fallback
            from the GPU shows.
    """

    digest: NonEmptyString
    weights_revision: NonEmptyString
    execution_providers: Annotated[list[str], Field(strict=True)]


@dataclass(frozen=True, slots=True)
class RunResult:
    """A run's result.

    Attributes:
        outputs: One output per declared output name.
        served_models: What each served model the analysis called ran, by
            served-model id; empty for a model-free analysis.
    """

    outputs: Annotated[dict[str, RunOutput], Field(strict=True)]
    served_models: Annotated[dict[str, ServedModelRun], Field(strict=True)] = field(
        default_factory=dict
    )


RESULT: TypeAdapter[RunResult] = TypeAdapter(RunResult)
