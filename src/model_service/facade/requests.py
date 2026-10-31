"""Public request bodies for running and scheduling builders.

The builder and model are named by the URL, `/v1/builders/{builder}/models/{model}`.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from model_service.builders.contracts import (
    BuilderInput,
    Deadline,
    InputAccessContext,
    Invocation,
    JsonValue,
)


class BuilderRequest(BaseModel):
    """Inputs and their access, and an optional output path.

    Processing settings are not part of the request; each attempt uses the
    builder's current defaults.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    inputs: Annotated[dict[str, BuilderInput], Field(strict=True)]
    input_access: InputAccessContext
    output_path: Annotated[str, Field(strict=True, min_length=1)] | None = None

    def invocation(self, deadline_at: Deadline) -> Invocation:
        return Invocation(
            inputs=self.inputs,
            resolved_configuration=None,
            input_access=self.input_access,
            deadline_at=deadline_at,
            output_path=self.output_path,
        )


class ImmediateBuilderRequest(BuilderRequest):
    deadline_at: Deadline


class ScheduledBuilderRequest(BuilderRequest):
    """A builder request plus a timeout; the `Idempotency-Key` header names it."""

    requested_timeout_seconds: Annotated[int, Field(strict=True, gt=0)] | None = None

    def builder_request(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json", exclude={"requested_timeout_seconds"})
