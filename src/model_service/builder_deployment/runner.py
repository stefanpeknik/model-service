"""Run one builder invocation behind validation, a deadline and resource cleanup.

The application checks the invocation and the model against the builder's
declaration, verifies the running model through its contract's client, opens the
inputs through the provider's access adapter, runs the builder before the
deadline, checks its outputs and always closes the access.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from model_service.builders.base import BuilderContext
from model_service.builders.declarations import BuilderInputs, BuilderOutputs
from model_service.builders.input_access import close_resource
from model_service.contracts.failures import (
    BuilderFailure,
    ContractViolation,
    FailureCode,
    InvalidInvocation,
)
from model_service.contracts.invocation import ExecutionContext
from model_service.contracts.json_values import is_json_value
from model_service.contracts.result import BuilderResult, InlineOutput, SharedArtifact
from model_service.deployed_models.model_clients import client_for, ray_handle


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.builders.base import Builder
    from model_service.builders.declarations import BuilderSpec
    from model_service.builders.input_access import InputAccess, InputAccessFactory
    from model_service.contracts.invocation import InputAccessKind, Invocation
    from model_service.contracts.model_declarations import ModelBinding
    from model_service.deployed_models.model_clients.base import HandleLookup


logger = logging.getLogger(__name__)


class BuilderApplication:
    """Runs one builder's invocations: validation, deadline, input access and cleanup."""

    def __init__(
        self,
        spec: BuilderSpec,
        builder: Builder[Any, Any, Any, Any],
        access_adapters: Mapping[InputAccessKind, InputAccessFactory],
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
        self._spec.check_invocation(invocation)
        self._spec.check_model(model)
        access: InputAccess | None = None
        primary_failure: BaseException | None = None
        try:
            remaining = (invocation.deadline_at - datetime.now(UTC)).total_seconds()
            timeout = asyncio.timeout(max(0, remaining))
            try:
                async with timeout:
                    # Before input access: a mismatched model fails without
                    # fetching inputs.
                    client = client_for(model, self._model_handles)
                    await client.verify()
                    access = await self._create_access(
                        invocation, execution_context or ExecutionContext(), model
                    )
                    outputs = await self._builder.run(
                        BuilderContext(
                            access=access,
                            model=client,
                            inputs=self._resolve_inputs(invocation, access),
                            configuration=self._spec.resolve_configuration(
                                invocation.resolved_configuration
                            ),
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
        """Resolve the supplied inputs to local files, as the builder's inputs model."""
        try:
            return self._spec.input_model(
                **{name: access.input_path(name) for name in invocation.inputs}
            )
        except ValidationError as error:
            # Names only: the paths may reveal provider storage.
            fields = sorted({".".join(map(str, e["loc"])) for e in error.errors()})
            raise InvalidInvocation(
                f"inputs are invalid: {', '.join(fields)}"
            ) from error

    def _output_paths(self, invocation: Invocation) -> dict[str, str]:
        """Map each artifact output to the relative path the builder writes it to."""
        paths = {
            name: output.default_path
            for name, output in self._spec.outputs.items()
            if output.default_path is not None
        }
        artifact = self._spec.placeable_output
        if invocation.output_path is not None and artifact is not None:
            paths[artifact] = invocation.output_path
        return paths

    async def _create_access(
        self,
        invocation: Invocation,
        execution_context: ExecutionContext,
        model: ModelBinding,
    ) -> InputAccess:
        """Create the input access through the adapter the invocation names."""
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
