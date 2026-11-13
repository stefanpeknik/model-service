"""A builder invocation: what is run, on which inputs, with which access, by when.

It crosses every process boundary (API, scheduler database, Ray builder
replica), so the types are strict and JSON-serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    TypeAdapter,
)
from pydantic.dataclasses import dataclass as validated_dataclass

from model_service.contracts.json_values import (
    JsonObject,
    NonEmptyString,
)


class InputKind(StrEnum):
    """How an invocation input is reached: a shared-storage path or a provider."""

    SHARED_PATH = "shared_path"
    PROVIDER = "provider"


@dataclass(frozen=True, slots=True)
class SharedPathInput:
    """A file below the invocation's shared-storage root."""

    path: NonEmptyString
    kind: Literal[InputKind.SHARED_PATH] = InputKind.SHARED_PATH


@dataclass(frozen=True, slots=True)
class ProviderInput:
    """An input the provider's access adapter materializes, e.g. an external job's input."""

    name: NonEmptyString
    kind: Literal[InputKind.PROVIDER] = InputKind.PROVIDER


type BuilderInput = Annotated[
    SharedPathInput | ProviderInput, Field(discriminator="kind")
]


# Which access adapter resolves an invocation's inputs and outputs; each
# provider names its own, the builder replica holds one adapter per kind.
type InputAccessKind = str


SHARED_STORAGE: InputAccessKind = "shared_storage"


@dataclass(frozen=True, slots=True)
class InputAccessContext:
    """Selects an access adapter and its public state."""

    kind: InputAccessKind
    state: JsonObject


@validated_dataclass(frozen=True, slots=True, config=ConfigDict(extra="forbid"))
class ExecutionContext:
    """Attempt identity and secret provider state, kept out of the stored invocation."""

    processing_attempt_id: NonEmptyString | None = None
    protected_context: JsonObject = field(default_factory=dict, repr=False)


def _iso_datetime(value: object) -> datetime:
    """Parse a deadline from an ISO 8601 string, passing datetimes through.

    Args:
        value: Datetime or ISO 8601 string.

    Returns:
        The parsed datetime.

    Raises:
        ValueError: The value is neither a datetime nor a valid ISO 8601 string.
    """
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError("deadline_at must be an ISO 8601 string")
    return datetime.fromisoformat(value)


type Deadline = Annotated[
    datetime,
    BeforeValidator(_iso_datetime),
    PlainSerializer(datetime.isoformat, return_type=str),
]


@dataclass(frozen=True, slots=True)
class Invocation:
    """One request to run a builder; stored durably for scheduled executions.

    `resolved_configuration` is None for standalone work, which uses the builder's
    current defaults when each attempt starts. A provider's admission may resolve
    it from an external job's configuration.
    """

    inputs: Annotated[dict[str, BuilderInput], Field(strict=True)]
    resolved_configuration: JsonObject | None
    input_access: InputAccessContext
    deadline_at: Deadline
    output_path: NonEmptyString | None = None


INVOCATION: TypeAdapter[Invocation] = TypeAdapter(Invocation)
