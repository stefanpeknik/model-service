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
from model_service.contracts.json_values import JsonObject


class BuilderRequest(BaseModel):
    """Inputs and their access, settings, and an optional output path.

    Attributes:
        inputs: The builder's inputs, by name.
        input_access: How the builder reaches them.
        configuration: Settings that differ from the builder's defaults; see the
            builder's configuration for the names, types and limits.
        output_path: Where the single artifact output is written.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    inputs: Annotated[dict[str, BuilderInput], Field(strict=True)]
    input_access: InputAccessContext
    configuration: Annotated[
        JsonObject | None,
        Field(
            description="Settings that differ from the builder's defaults, by "
            "name; GET /v1/builders lists each builder's settings with their "
            "defaults."
        ),
    ] = None
    output_path: Annotated[str, Field(strict=True, min_length=1)] | None = None

    def invocation(self, deadline_at: Deadline) -> Invocation:
        """Build the invocation of this request; each attempt applies its settings.

        Args:
            deadline_at: When the execution must finish.

        Returns:
            The invocation.
        """
        return Invocation(
            inputs=self.inputs,
            configuration=self.configuration,
            input_access=self.input_access,
            deadline_at=deadline_at,
            output_path=self.output_path,
        )
