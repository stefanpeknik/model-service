"""Public request bodies for running and scheduling builders.

The builder and model are named by the URL, `/v1/builders/{builder}/models/{model}`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from pydantic import Field

from model_service.builders.contracts import Deadline
from model_service.providers.standalone import BuilderRequest


if TYPE_CHECKING:
    from model_service.builders.contracts import JsonValue


class ImmediateBuilderRequest(BuilderRequest):
    deadline_at: Deadline


class ScheduledBuilderRequest(BuilderRequest):
    """A builder request plus a timeout; the `Idempotency-Key` header names it."""

    requested_timeout_seconds: Annotated[int, Field(strict=True, gt=0)] | None = None

    def builder_request(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json", exclude={"requested_timeout_seconds"})
