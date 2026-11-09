"""Run one builder invocation behind validation, a deadline and resource cleanup.

Access adapters turn provider-specific inputs into local paths, so builders only
see `BuilderContext`.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from model_service.builders.artifacts import atomic_artifact
from model_service.builders.cleanup import close_resource
from model_service.builders.contracts import (
    BuilderFailure,
    BuilderInputs,
    BuilderOutputs,
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
from model_service.builders.model_clients import client_for, ray_handle


if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterator, Mapping

    from pydantic import BaseModel

    from model_service.builders.contracts import (
        InputAccessKind,
        Invocation,
        ModelBinding,
    )
    from model_service.builders.model_clients import HandleLookup, ModelClient
    from model_service.builders.spec import BuilderSpec


logger = logging.getLogger(__name__)


class InputAccess(ABC):
    @abstractmethod
    def input_path(self, name: str) -> Path:
        """Resolve a named invocation input to a local file.

        Args:
            name: Input name from the builder's declaration.

        Returns:
            Local path the builder can read the input from.
        """
        ...

    @abstractmethod
    def output_path(self, relative_path: str) -> Path:
        """Resolve an output's relative path to the local file to write.

        Args:
            relative_path: Output path relative to the invocation's output location.

        Returns:
            Local path the builder writes the artifact to.
        """
        ...

    @abstractmethod
    def is_visible_artifact(self, path: Path) -> bool:
        """Check that a returned artifact is a file this access exposes to callers.

        Args:
            path: Artifact path a builder returned in its result.

        Returns:
            True when the path is a file inside the access's visible output area.
        """
        ...

    @abstractmethod
    async def aclose(self) -> None:
        """Release what the access holds, such as fetched inputs or temporary files."""
        ...


class AccessAdapter(ABC):
    # The model tells an adapter that fetches slides which resolution the builder reads.
    @abstractmethod
    def create(
        self,
        invocation: Invocation,
        execution_context: ExecutionContext,
        model: ModelBinding,
    ) -> InputAccess | Awaitable[InputAccess]:
        """Create the input access for one invocation attempt.

        Args:
            invocation: Invocation whose inputs and outputs the access resolves.
            execution_context: Attempt identity and protected provider state.
            model: Model binding; tells an adapter that fetches slides which
                resolution the builder reads.

        Returns:
            The access, or an awaitable resolving to it.
        """
        ...


@dataclass(frozen=True, slots=True)
class BuilderContext[K: ModelClient[Any], I: BuilderInputs, C: BaseModel]:
    """What one builder run gets: its model, inputs and configuration, typed.

    The application checks the model against the builder's contract and its
    running deployment against the catalog, resolves the inputs to local files
    and the configuration with the builder's own classes before building it, so
    a builder can rely on these types without checking them again.
    """

    access: InputAccess
    # The verified client of the model's contract; `model.declaration` is its
    # catalog entry.
    model: K
    inputs: I
    configuration: C
    deadline_at: datetime
    output_paths: Mapping[str, str]

    def output_path(self, output: str) -> Path:
        """Return where to write a declared artifact output.

        Uses the invocation's `output_path` override when it set one, otherwise the
        output's default path. `write_artifact` is the usual way to write one.

        Args:
            output: Output name from the builder's declaration.

        Returns:
            Local path for the output file.

        Raises:
            KeyError: The output has no path, i.e. it is not an artifact output.
        """
        return self.access.output_path(self.output_paths[output])

    @contextmanager
    def write_artifact(self, output: str) -> Iterator[PendingArtifact]:
        """Write an artifact output atomically.

        The builder writes `pending.path`, a temporary file next to the
        destination; on success it replaces the destination, and on failure it
        is removed, so a failed run leaves no partial artifact. After the block,
        `pending.artifact` is the output to return.

        Args:
            output: Output name from the builder's declaration.

        Yields:
            The pending artifact.

        Raises:
            InvalidInvocation: The destination is one of the inputs.
            RuntimeError: The block did not write a regular file.
        """
        destination = self.output_path(output)
        with atomic_artifact(destination, inputs=self.inputs.paths()) as temporary:
            yield PendingArtifact(path=temporary, destination=destination)


@dataclass(frozen=True, slots=True)
class PendingArtifact:
    """An artifact being written by `BuilderContext.write_artifact`."""

    # Where the builder writes; closed before the block ends.
    path: Path
    destination: Path

    @property
    def artifact(self) -> SharedArtifact:
        """The finished artifact, to return in the builder's outputs."""
        return SharedArtifact(path=str(self.destination))


class Builder[
    K: ModelClient[Any],
    I: BuilderInputs,
    C: BaseModel,
    R: BuilderOutputs,
](ABC):
    """Produces a builder's outputs from a slide and a model.

    `K` is the model client of the contract it accepts; `I`, `C` and `R` are its
    declared inputs, configuration and outputs models.
    """

    @abstractmethod
    async def run(self, context: BuilderContext[K, I, C]) -> R:
        """Produce the builder's outputs for one invocation.

        Args:
            context: Verified model client, resolved inputs, configuration and
                deadline.

        Returns:
            The builder's outputs model, one field per declared output.
        """
        ...


def validate_invocation(
    spec: BuilderSpec, invocation: Invocation, *, now: datetime | None = None
) -> None:
    """Check an invocation against a builder's declaration before running it.

    Args:
        spec: Declaration of the builder to run.
        invocation: Invocation to check.
        now: Time to compare the deadline against; defaults to the current UTC time.

    Raises:
        InvalidInvocation: The deadline lacks a timezone or has passed, inputs are
            unknown or missing, the configuration is invalid, or `output_path` is
            set for a builder without exactly one artifact output.
    """
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
    """Check that a model's inference contract is the one the builder accepts.

    Args:
        spec: Declaration of the builder to run.
        model: Model binding resolved for the attempt.

    Raises:
        IncompatibleModelError: The model's contract differs from `spec.accepts`.
    """
    if model.contract != spec.accepts:
        raise IncompatibleModelError(
            f"model {model.id!r} ({model.contract}) is incompatible with "
            f"builder {spec.id!r} ({spec.accepts})"
        )


def _artifact_output(spec: BuilderSpec) -> str | None:
    """Find the single artifact output an invocation's `output_path` can place.

    Args:
        spec: Builder declaration.

    Returns:
        The name of the only output with a default path, or None when there are
        none or several.
    """
    names = [name for name, output in spec.outputs.items() if output.default_path]
    return names[0] if len(names) == 1 else None


class BuilderApplication:
    def __init__(
        self,
        spec: BuilderSpec,
        builder: Builder[Any, Any, Any, Any],
        access_adapters: Mapping[InputAccessKind, AccessAdapter],
        *,
        cleanup_timeout_seconds: float = 10,
        model_handles: HandleLookup = ray_handle,
    ) -> None:
        """Bind a builder to its declaration and access adapters.

        Args:
            spec: Declaration the invocations and results are checked against.
            builder: Builder implementation to run.
            access_adapters: Adapter per input access kind an invocation may name.
            cleanup_timeout_seconds: Time allowed for closing the input access.
            model_handles: Looks up a model's deployment handle by id; replaceable
                in tests.
        """
        self._spec = spec
        self._builder = builder
        self._access_adapters = dict(access_adapters)
        self._cleanup_timeout_seconds = cleanup_timeout_seconds
        self._model_handles = model_handles

    async def run(
        self,
        invocation: Invocation,
        model: ModelBinding,
        execution_context: ExecutionContext | None = None,
    ) -> BuilderResult:
        """Validate, run the builder before the deadline, check its result, and clean up.

        The input access is always closed. A cleanup failure is raised only when the
        run itself succeeded; otherwise it is logged and the original error wins.

        Args:
            invocation: Invocation to run.
            model: Model binding resolved for this attempt.
            execution_context: Attempt identity and protected provider state; an
                empty context when omitted.

        Returns:
            The builder's validated result.

        Raises:
            InvalidInvocation: The invocation, its access kind or its configuration is
                invalid, or the model is incompatible.
            ContractViolation: The builder returned outputs that differ from its
                declaration.
            BuilderFailure: The running model differs from its catalog entry, the
                deadline passed (both not retryable), the builder or the cleanup
                failed unexpectedly (retryable), or the model client or builder
                raised one.
        """
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
                    client = client_for(model, self._model_handles)
                    await client.verify()
                    outputs = await self._builder.run(
                        BuilderContext(
                            access=access,
                            model=client,
                            inputs=self._resolve_inputs(invocation, access),
                            configuration=self._spec.resolve_configuration(
                                invocation.resolved_configuration
                            ),
                            deadline_at=invocation.deadline_at,
                            output_paths=self._output_paths(invocation),
                        )
                    )
                    result = self._validate_result(outputs, access)
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

    def _resolve_inputs(
        self, invocation: Invocation, access: InputAccess
    ) -> BuilderInputs:
        """Resolve the supplied inputs to local files, as the builder's inputs model.

        Args:
            invocation: Invocation naming the supplied inputs.
            access: Access resolving each input to a local file.

        Returns:
            The builder's `BuilderInputs` subclass; omitted optional inputs are None.
        """
        return self._spec.input_model(
            **{name: access.input_path(name) for name in invocation.inputs}
        )

    def _output_paths(self, invocation: Invocation) -> dict[str, str]:
        """Map each artifact output to the path the builder writes it to.

        Args:
            invocation: Invocation whose `output_path` may override the default path
                of the single artifact output.

        Returns:
            Output name to relative path, for outputs that have one.
        """
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
        """Create the input access through the adapter the invocation names.

        Args:
            invocation: Invocation whose `input_access.kind` selects the adapter.
            execution_context: Attempt identity and protected provider state.
            model: Model binding passed on to the adapter.

        Returns:
            The created input access.

        Raises:
            InvalidInvocation: No adapter is registered for the access kind.
        """
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
        """Close the input access within the cleanup timeout.

        Args:
            access: Access to close.
            primary_failure: Error the run already raised, if any; when set, a
                cleanup failure is only logged.

        Raises:
            asyncio.CancelledError: The cleanup was cancelled.
            BuilderFailure: Cleanup failed and the run had not failed already.
        """
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

    def _validate_result(
        self, outputs: BuilderOutputs, access: InputAccess
    ) -> BuilderResult:
        """Check a builder's result against its output declaration.

        A builder need not repeat an artifact's media type: the declared one is
        set on it here, and one it does give must match.

        Args:
            outputs: What the builder returned.
            access: Access used to check that artifacts are visible files.

        Returns:
            The result, with each artifact carrying its declared media type.

        Raises:
            ContractViolation: The result is not a `BuilderOutputs`, its output names
                differ from the declaration, an output has the wrong type or kind, an
                artifact is not a visible file or has the wrong media type, or an
                inline value is not JSON.
        """
        if not isinstance(outputs, BuilderOutputs):
            raise ContractViolation("builder returned an invalid result")
        result = outputs.result()
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
                if output.media_type not in {None, spec.media_type}:
                    raise ContractViolation(
                        f"artifact {name!r} has the wrong media type"
                    )
            elif not isinstance(output, InlineOutput) or not is_json_value(
                output.value
            ):
                raise ContractViolation(f"inline output {name!r} is not JSON")
        return replace(
            result,
            outputs={
                name: (
                    replace(output, media_type=self._spec.outputs[name].media_type)
                    if isinstance(output, SharedArtifact)
                    else output
                )
                for name, output in result.outputs.items()
            },
        )
