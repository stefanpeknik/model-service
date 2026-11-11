"""A request to run a builder: inputs, their access, configuration and output path."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from model_service.contracts.invocation import (
    BuilderInput,
    Deadline,
    InputAccessContext,
    Invocation,
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
