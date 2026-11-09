"""Shared fixtures for builder declaration tests: test specs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, create_model

from model_service.builders.contracts import (
    Artifact,
    BuilderInputs,
    BuilderOutput,
    BuilderOutputs,
    BuilderResult,
    InferenceContract,
    InlineOutput,
    Input,
    OutputKind,
    SharedArtifact,
)
from model_service.builders.spec import BuilderSpec, InputSpec, OutputSpec


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from model_service.builders.base import BuilderContext


class EmptyConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StrideConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stride_fraction: float = 0.5


def spec(
    *,
    inputs: Mapping[str, InputSpec] | None = None,
    outputs: Mapping[str, OutputSpec] | None = None,
    configuration: type[BaseModel] = EmptyConfiguration,
    accepts: InferenceContract = InferenceContract.SPATIAL_SCORES,
) -> BuilderSpec:
    return BuilderSpec(
        id="test-builder",
        accepts=accepts,
        input_model=inputs_model(inputs or {}),
        output_model=outputs_model(outputs or {}),
        configuration=configuration,
        implementation="tests.support.builder_fixtures:FunctionBuilder",
    )


def inputs_model(inputs: Mapping[str, InputSpec]) -> type[BuilderInputs]:
    """A builder inputs model declaring these inputs, for test specs."""
    fields: dict[str, Any] = {}
    for name, item in inputs.items():
        marker = Input(item.media_type)
        if item.required:
            fields[name] = (Annotated[Path, marker], ...)
        else:
            fields[name] = (Annotated[Path | None, marker], None)
    return create_model("TestInputs", __base__=BuilderInputs, **fields)


def outputs_model(outputs: Mapping[str, OutputSpec]) -> type[BuilderOutputs]:
    """A builder outputs model declaring these outputs, for test specs."""
    fields: dict[str, Any] = {}
    for name, output in outputs.items():
        if output.kind is OutputKind.INLINE:
            fields[name] = (InlineOutput, ...)
        else:
            artifact = Artifact(output.media_type, output.default_path)
            fields[name] = (Annotated[SharedArtifact, artifact], ...)
    return create_model("TestOutputs", __base__=BuilderOutputs, **fields)


class FunctionBuilder:
    def __init__(
        self, function: Callable[[BuilderContext], Awaitable[BuilderResult]]
    ) -> None:
        self._function = function

    async def run(self, context: BuilderContext[Any, Any]) -> BuilderOutputs:
        """Return the function's result as outputs, without validating it.

        Unvalidated, so tests can return wrong names and kinds.
        """
        result = await self._function(context)
        if not isinstance(result, BuilderResult):
            return result  # type: ignore[return-value]
        model = create_model(
            "Returned",
            __base__=BuilderOutputs,
            **dict.fromkeys(result.outputs, (BuilderOutput, ...)),  # type: ignore[call-overload]
        )
        return model.model_construct(**result.outputs)
