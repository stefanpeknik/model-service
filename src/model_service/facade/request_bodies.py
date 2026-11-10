"""Public request bodies for running and scheduling builders.

The builder and model are named by the URL, `/v1/builders/{builder}/models/{model}`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field

from model_service.wire.invocation import (
    BuilderInput,
    Deadline,
    InputAccessContext,
    Invocation,
)


if TYPE_CHECKING:
    from model_service.wire.json import JsonValue


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
        """Build the invocation of this request; the attempt resolves the configuration.

        Args:
            deadline_at: When the execution must finish.

        Returns:
            The invocation, with no resolved configuration.
        """
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
        """Serialize the builder request without the scheduling timeout.

        Returns:
            The JSON body stored with the execution.
        """
        return self.model_dump(mode="json", exclude={"requested_timeout_seconds"})
