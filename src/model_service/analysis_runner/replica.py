"""The analysis-runner replica core: one run request, checked and executed.

Per request, in order, before the deadline:

1. the resolved document is read back and its digest checked;
2. each served model it names is asked for `describe()`, and a digest that
   differs from the document's, or weights other than the ones its file pins
   (a Hugging Face commit, an MLflow `sha256`), is `model_contract_mismatch`;
3. the provider's input access opens the inputs and says where outputs
   go: below the request's shared-storage root, which must lie within the
   replica's allowed roots, or in an EMPAIA job's workspace, with each input
   converted by its recorded EAD type;
4. `run_analysis` runs it, with progress passed to the access;
5. the result names each output's file and its format's media type.

No Ray here: the Serve wrapper (`serve.py`) gives it Ray handles, tests give
it fakes.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from model_service.analysis_runner.models import RayServedModel
from model_service.analysis_runner.progress import reporting_progress
from model_service.analysis_runner.run import Concurrency, run_analysis
from model_service.catalog.analyses import ResolvedAnalysisInvalid, load_resolved
from model_service.contracts.failures import (
    FailureCode,
    InvalidInvocation,
    RunFailure,
)
from model_service.contracts.invocation import ExecutionContext
from model_service.contracts.log_context import logging_execution
from model_service.contracts.result import RunResult, SharedArtifact
from model_service.formats.known import get_format
from model_service.providers.input_access import close_resource


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping
    from pathlib import Path

    from model_service.analysis_runner.models import DescribedModel
    from model_service.analysis_runner.settings import ReplicaSettings
    from model_service.catalog.analyses import Analysis, ResolvedAnalysis
    from model_service.contracts.analysis_http import AnalysisRunRequest
    from model_service.contracts.invocation import InputAccessKind, Invocation
    from model_service.providers.input_access import InputAccess


logger = logging.getLogger(__name__)


class AccessFactory(Protocol):
    """Creates the input access of one attempt: shared storage or a provider's."""

    def create(
        self, invocation: Invocation, execution_context: ExecutionContext
    ) -> InputAccess | Awaitable[InputAccess]:
        """Create the access for one attempt.

        Args:
            invocation: Names the inputs, the root and the output path.
            execution_context: Attempt identity, which scopes outputs.

        Returns:
            The access, or an awaitable giving it.
        """
        ...


class AnalysisReplica:
    """Runs resolved analyses for one runner replica."""

    def __init__(
        self,
        settings: ReplicaSettings,
        *,
        models: Callable[[str], DescribedModel] = RayServedModel,
        access: Mapping[InputAccessKind, AccessFactory] | None = None,
        progress_interval_seconds: float = 10,
        cleanup_timeout_seconds: float = 10,
    ) -> None:
        """Take the replica's limits and allowed roots.

        Args:
            settings: Threads and tiles in flight per run, and the
                shared-storage roots requests may name.
            models: Gives a served model's handle by ID.
            access: The input access by kind; by default every provider's.
            progress_interval_seconds: Time between two progress sends.
            cleanup_timeout_seconds: Time allowed for closing the access.
        """
        self._concurrency = Concurrency(
            read_threads=settings.num_threads,
            in_flight=settings.max_concurrent_tasks,
        )
        self._access = access if access is not None else _provider_access(settings)
        self._models = models
        self._progress_interval_seconds = progress_interval_seconds
        self._cleanup_timeout_seconds = cleanup_timeout_seconds

    async def run(self, request: AnalysisRunRequest) -> RunResult:
        """Check and run one request before its deadline.

        Args:
            request: The resolved document, the invocation and the attempt.

        Returns:
            Each output's file and media type.

        Raises:
            InvalidInvocation: The document was changed or doesn't resolve,
                or the inputs, root or output path are invalid.
            RunFailure: `model_contract_mismatch` (a served model runs
                another declaration), `run_deadline_exceeded`, or what
                the run raised (`model_failed`, `invalid_model_output`,
                `run_internal_error`).
        """
        with logging_execution(request.execution_context.execution_id):
            try:
                resolved = load_resolved(request.analysis)
            except ResolvedAnalysisInvalid as error:
                raise InvalidInvocation(str(error)) from error
            remaining = (
                request.invocation.deadline_at - datetime.now(UTC)
            ).total_seconds()
            timeout = asyncio.timeout(max(0, remaining))
            try:
                async with timeout:
                    return await self._run(resolved, request)
            except TimeoutError as error:
                if not timeout.expired():
                    raise
                raise RunFailure(
                    code=FailureCode.DEADLINE_EXCEEDED,
                    safe_message="analysis deadline was exceeded",
                    retryable=False,
                ) from error

    async def _run(
        self, resolved: ResolvedAnalysis, request: AnalysisRunRequest
    ) -> RunResult:
        """Verify the models, open the access, run, and close the access.

        Args:
            resolved: The analysis read back from the request's document.
            request: The run request.

        Returns:
            Each output's file and media type.

        Raises:
            InvalidInvocation: For an unsupported access kind or inputs and
                outputs that don't fit the analysis.
            RunFailure: From the model checks and the run.
        """
        analysis = resolved.analysis
        models = {
            model_id: await self._verified(
                model_id, digest, analysis.models[model_id].weights.expected_revision()
            )
            for model_id, digest in resolved.model_digests.items()
        }
        invocation = request.invocation
        factory = self._access.get(invocation.input_access.kind)
        if factory is None:
            raise InvalidInvocation("unsupported input access kind")
        created = factory.create(invocation, request.execution_context)
        access = await created if inspect.isawaitable(created) else created
        try:
            inputs = _inputs(analysis, invocation, access)
            outputs = _outputs(analysis, invocation, access)
            async with reporting_progress(
                access.report_progress, interval_seconds=self._progress_interval_seconds
            ) as report_progress:
                written = await run_analysis(
                    analysis,
                    inputs,
                    outputs,
                    models,
                    concurrency=self._concurrency,
                    report_progress=report_progress,
                )
        finally:
            await self._close(access)
        formats = {s.options.output: get_format(s.format) for s in analysis.sinks}
        return RunResult(
            outputs={
                name: SharedArtifact(str(path), formats[name].media_type)
                for name, path in written.items()
            }
        )

    async def _verified(
        self, model_id: str, digest: str, weights: str | None
    ) -> DescribedModel:
        """A served model's handle, after its `describe()` matched the document.

        Args:
            model_id: The served model.
            digest: Its digest in the resolved document.
            weights: The weights revision its file pins, or None if it pins none.

        Returns:
            The handle.

        Raises:
            RunFailure: `model_failed` (retryable) if `describe()` fails;
                `model_contract_mismatch` if the digest or pinned weights differ.
        """
        handle = self._models(model_id)
        try:
            running = await handle.describe()
        except Exception as error:
            raise RunFailure(
                code=FailureCode.MODEL_FAILED,
                safe_message=f"model {model_id!r} could not be reached",
                retryable=True,
            ) from error
        if not isinstance(running, dict) or running.get("digest") != digest:
            raise RunFailure(
                code=FailureCode.MODEL_CONTRACT_MISMATCH,
                safe_message=(
                    f"model {model_id!r} is running a declaration that differs "
                    "from the one the analysis was admitted with"
                ),
                retryable=False,
            )
        if weights is not None and running.get("weights_revision") != weights:
            raise RunFailure(
                code=FailureCode.MODEL_CONTRACT_MISMATCH,
                safe_message=(
                    f"model {model_id!r} is running other weights than its file pins"
                ),
                retryable=False,
            )
        return handle

    async def _close(self, access: InputAccess) -> None:
        """Close the access, logging a failure rather than hiding the run's outcome.

        Args:
            access: The run's input access.

        Raises:
            asyncio.CancelledError: If the close is cancelled.
        """
        try:
            await close_resource(
                access.aclose, timeout_seconds=self._cleanup_timeout_seconds
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning("input access cleanup failed: %s", type(error).__name__)


def _provider_access(
    settings: ReplicaSettings,
) -> dict[InputAccessKind, AccessFactory]:
    """Every provider's input access, by the kind each serves.

    Args:
        settings: The replica's settings, which the providers read.

    Returns:
        Access factory by input access kind.
    """
    from model_service.providers.registry import PROVIDERS

    access: dict[InputAccessKind, AccessFactory] = {}
    for provider in PROVIDERS:
        access.update(provider.input_access(settings))
    return access


def _inputs(
    analysis: Analysis, invocation: Invocation, access: InputAccess
) -> dict[str, Path]:
    """Resolve each given input below the root; names the analysis doesn't read are refused.

    Args:
        analysis: The analysis.
        invocation: The run's invocation.
        access: The run's input access.

    Returns:
        Local path by input name.

    Raises:
        InvalidInvocation: For an input the analysis doesn't read.
    """
    unknown = sorted(set(invocation.inputs) - set(analysis.inputs))
    if unknown:
        raise InvalidInvocation(
            f"inputs of {analysis.id}: {', '.join(sorted(analysis.inputs))}; "
            f"unknown {unknown}"
        )
    return {name: access.input_path(name) for name in invocation.inputs}


def _outputs(
    analysis: Analysis, invocation: Invocation, access: InputAccess
) -> dict[str, Path]:
    """Where each output goes: `output_path` for an analysis with one output.

    Without `output_path`, each output is `<output>.<format extension>` at
    the root.

    Args:
        analysis: The analysis.
        invocation: The run's invocation.
        access: The run's input access, which resolves the paths.

    Returns:
        Local path by output name.

    Raises:
        InvalidInvocation: For `output_path` with more than one output.
    """
    paths = {
        s.options.output: f"{s.options.output}.{get_format(s.format).extension}"
        for s in analysis.sinks
    }
    if invocation.output_path is not None:
        if len(paths) != 1:
            raise InvalidInvocation(
                f"output_path needs an analysis with one output; {analysis.id} "
                f"writes {', '.join(sorted(paths))}"
            )
        paths = dict.fromkeys(paths, invocation.output_path)
    return {name: access.output_path(path) for name, path in paths.items()}
