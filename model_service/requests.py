"""Public request bodies for running and scheduling builders."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from builders.contracts import (
    BuilderInput,
    Deadline,
    InputAccessContext,
    Invocation,
)


class BuilderRequest(BaseModel):
    """Inputs and their access, the selected model and an optional output path.

    Processing settings are not part of the request; each attempt uses the
    builder's current defaults.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: Annotated[str, Field(strict=True, min_length=1)]
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
