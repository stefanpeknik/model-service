"""A builder's result as it is sent and stored: one output per declared name."""

from __future__ import annotations

from dataclasses import dataclass
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
    """How a builder output is returned: inline or as a shared artifact."""

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


type BuilderOutput = Annotated[
    InlineOutput | SharedArtifact, Field(discriminator="kind")
]


@dataclass(frozen=True, slots=True)
class BuilderResult:
    """A builder run's result.

    Attributes:
        outputs: One output per declared output name.
    """

    outputs: Annotated[dict[str, BuilderOutput], Field(strict=True)]


RESULT: TypeAdapter[BuilderResult] = TypeAdapter(BuilderResult)
