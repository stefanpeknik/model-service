"""Run one builder invocation behind validation, a deadline and resource cleanup.

Access adapters turn provider-specific inputs into local paths, so builders only
see `BuilderContext`.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from model_service.builders.cleanup import close_resource
from model_service.builders.contracts import (
    BuilderFailure,
    BuilderResult,
    ContractViolation,
    ExecutionContext,
    FailureCode,
    IncompatibleModelError,
    InlineOutput,
    InvalidInvocation,
    SharedArtifact,
    is_json_value,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping

    from pydantic import BaseModel

    from model_service.builders.contracts import (
        InputAccessKind,
        Invocation,
        ModelBinding,
    )
    from model_service.builders.registry import BuilderSpec


logger = logging.getLogger(__name__)


class InputAccess(Protocol):
    def input_path(self, name: str) -> Path: ...

    def output_path(self, relative_path: str) -> Path: ...

    def is_visible_artifact(self, path: Path) -> bool: ...

    async def aclose(self) -> None: ...


class AccessAdapter(Protocol):
    # The model tells an adapter that fetches slides which resolution the builder reads.
    def create(
        self,
        invocation: Invocation,
        execution_context: ExecutionContext,
        model: ModelBinding,
    ) -> InputAccess | Awaitable[InputAccess]: ...


@dataclass(frozen=True, slots=True)
class BuilderContext:
    access: InputAccess
    input_names: frozenset[str]
    configuration: BaseModel
    model: ModelBinding
    deadline_at: datetime
    output_paths: Mapping[str, str]

    def input_path(self, name: str) -> Path:
        return self.access.input_path(name)

    def optional_input_path(self, name: str) -> Path | None:
        return self.access.input_path(name) if name in self.input_names else None

    def output_path(self, output: str) -> Path:
        return self.access.output_path(self.output_paths[output])


class Builder(Protocol):
    async def run(self, context: BuilderContext) -> BuilderResult: ...


def validate_invocation(
    spec: BuilderSpec, invocation: Invocation, *, now: datetime | None = None
) -> None:
    deadline = invocation.deadline_at
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise InvalidInvocation("deadline_at must include a timezone")
    if deadline <= (now or datetime.now(UTC)):
        raise InvalidInvocation("deadline_at must be in the future")
    unknown = invocation.inputs.keys() - spec.inputs.keys()
    if unknown:
        raise InvalidInvocation(f"unknown inputs: {', '.join(sorted(unknown))}")
    missing = {
        name
        for name, input_spec in spec.inputs.items()
        if input_spec.required and name not in invocation.inputs
    }
    if missing:
        raise InvalidInvocation(f"missing inputs: {', '.join(sorted(missing))}")
    if invocation.resolved_configuration is not None:
        spec.resolve_configuration(invocation.resolved_configuration)
    if invocation.output_path is not None and _artifact_output(spec) is None:
        raise InvalidInvocation(f"{spec.id} has no artifact output to place")


def validate_model(spec: BuilderSpec, model: ModelBinding) -> None:
    if model.contract != spec.accepts:
        raise IncompatibleModelError(
            f"model {model.id!r} ({model.contract}) is incompatible with "
            f"builder {spec.id!r} ({spec.accepts})"
        )


def _artifact_output(spec: BuilderSpec) -> str | None:
    names = [name for name, output in spec.outputs.items() if output.default_path]
    return names[0] if len(names) == 1 else None


class BuilderApplication:
    def __init__(
        self,
        spec: BuilderSpec,
        builder: Builder,
        access_adapters: Mapping[InputAccessKind, AccessAdapter],
        *,
        cleanup_timeout_seconds: float = 10,
    ) -> None:
        self._spec = spec
        self._builder = builder
        self._access_adapters = dict(access_adapters)
        self._cleanup_timeout_seconds = cleanup_timeout_seconds

    async def run(
        self,
        invocation: Invocation,
        model: ModelBinding,
        execution_context: ExecutionContext | None = None,
    ) -> BuilderResult:
        validate_invocation(self._spec, invocation)
        validate_model(self._spec, model)
        access: InputAccess | None = None
        primary_failure: BaseException | None = None
        try:
            remaining = (invocation.deadline_at - datetime.now(UTC)).total_seconds()
            timeout = asyncio.timeout(max(0, remaining))
            try:
                async with timeout:
                    access = await self._create_access(
                        invocation, execution_context or ExecutionContext(), model
                    )
                    result = await self._builder.run(
                        BuilderContext(
                            access=access,
                            input_names=frozenset(invocation.inputs),
                            configuration=self._spec.resolve_configuration(
                                invocation.resolved_configuration
                            ),
                            model=model,
                            deadline_at=invocation.deadline_at,
                            output_paths=self._output_paths(invocation),
                        )
                    )
                    self._validate_result(result, access)
            except TimeoutError as error:
                if not timeout.expired():
                    raise
                raise BuilderFailure(
                    code=FailureCode.DEADLINE_EXCEEDED,
                    safe_message="builder deadline was exceeded",
                    retryable=False,
                ) from error
        except (InvalidInvocation, ContractViolation, BuilderFailure) as error:
            primary_failure = error
            raise
        except Exception as error:
            logger.exception("builder %s failed unexpectedly", self._spec.id)
            primary_failure = BuilderFailure(
                code=FailureCode.INTERNAL_ERROR,
                safe_message="builder failed unexpectedly",
                retryable=True,
            )
            raise primary_failure from error
        except BaseException as error:
            primary_failure = error
            raise
        finally:
            if access is not None:
                await self._close(access, primary_failure)
        return result

    def _output_paths(self, invocation: Invocation) -> dict[str, str]:
        paths = {
            name: output.default_path
            for name, output in self._spec.outputs.items()
            if output.default_path is not None
        }
        artifact = _artifact_output(self._spec)
        if invocation.output_path is not None and artifact is not None:
            paths[artifact] = invocation.output_path
        return paths

    async def _create_access(
        self,
        invocation: Invocation,
        execution_context: ExecutionContext,
        model: ModelBinding,
    ) -> InputAccess:
        adapter = self._access_adapters.get(invocation.input_access.kind)
        if adapter is None:
            raise InvalidInvocation("unsupported input access kind")
        access = adapter.create(invocation, execution_context, model)
        if inspect.isawaitable(access):
            return await access
        return access

    async def _close(
        self, access: InputAccess, primary_failure: BaseException | None
    ) -> None:
        try:
            await close_resource(
                access.aclose, timeout_seconds=self._cleanup_timeout_seconds
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if primary_failure is None:
                raise BuilderFailure(
                    code=FailureCode.INTERNAL_ERROR,
                    safe_message="builder resource cleanup failed",
                    retryable=True,
                ) from error
            logger.warning("builder resource cleanup failed: %s", type(error).__name__)

    def _validate_result(self, result: BuilderResult, access: InputAccess) -> None:
        if not isinstance(result, BuilderResult):
            raise ContractViolation("builder returned an invalid result")
        if result.outputs.keys() != self._spec.outputs.keys():
            raise ContractViolation("outputs do not match the builder declaration")
        for name, spec in self._spec.outputs.items():
            output = result.outputs[name]
            if not isinstance(output, InlineOutput | SharedArtifact):
                raise ContractViolation(f"output {name!r} is not a builder output")
            if output.kind != spec.kind:
                raise ContractViolation(f"output {name!r} has the wrong kind")
            if isinstance(output, SharedArtifact):
                if not access.is_visible_artifact(Path(output.path)):
                    raise ContractViolation(f"artifact {name!r} is not a visible file")
                if spec.media_type is not None and output.media_type != spec.media_type:
                    raise ContractViolation(
                        f"artifact {name!r} has the wrong media type"
                    )
            elif not isinstance(output, InlineOutput) or not is_json_value(
                output.value
            ):
                raise ContractViolation(f"inline output {name!r} is not JSON")
