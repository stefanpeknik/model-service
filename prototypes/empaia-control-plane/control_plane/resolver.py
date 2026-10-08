"""Resolve admitted jobs into local storage locators and immutable worker specifications.

Historical SQLite control-plane prototype. The current production scheduler lives in
execution_scheduler/ and uses PostgreSQL; these lifecycle rules are not its
specification.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from control_plane.app_service import (
    AppServiceApplicationError,
    AppServiceInfrastructureError,
    AppServiceTransientError,
)
from control_plane.domain import (
    Execution,
    ExecutionSpec,
    FailureCategory,
    InvalidTransitionError,
    JsonValue,
)
from control_plane.storage import InvalidStorageInputError, StorageUnavailableError


if TYPE_CHECKING:
    from control_plane.app_service import DirectAccess, JobConfiguration
    from control_plane.storage import StorageLocatorResolver
    from control_plane.store import SQLiteExecutionStore


logger = logging.getLogger(__name__)


class ResolverAppService(Protocol):
    """Define configuration, direct-access, and failure-report calls needed by prototype resolution."""

    def get_configuration(self, job_id: str) -> JobConfiguration:
        """Fetch the external job configuration for allowed model-parameter selection.

        Args:
            job_id: External EMPAIA job identifier.

        Returns:
            JobConfiguration: Validated global and customer provider configuration
                sections.
        """
        ...

    def get_direct_access(self, job_id: str, input_name: str) -> DirectAccess:
        """Resolve one job input to provider WSI storage metadata.

        Args:
            job_id: External EMPAIA job identifier.
            input_name: Named external job input to resolve.

        Returns:
            DirectAccess: Validated provider shared path and direct-access support
                metadata.
        """
        ...

    def report_failure(self, job_id: str, user_message: str) -> None:
        """Report a safe unsupported-job reason to AppService before terminal settlement.

        Args:
            job_id: External EMPAIA job identifier.
            user_message: Safe external job failure message, bounded by the provider
                contract.

        Returns:
            None: No return value.
        """
        ...


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    """Count candidates made ready, deferred, or failed during one resolver pass.

    Attributes:
        ready: Count of candidates successfully resolved.
        deferred: Count of candidates awaiting another resolution pass.
        failed: Count of candidates settled as failures.
    """

    ready: int = 0
    deferred: int = 0
    failed: int = 0


class EmpaiaExecutionResolver:
    """Turns durable admissions into immutable, credential-free worker specs."""

    def __init__(
        self,
        store: SQLiteExecutionStore,
        app_service: ResolverAppService,
        storage: StorageLocatorResolver,
        retry_seconds: float,
    ) -> None:
        """Require a positive retry delay and retain the store, synchronous AppService boundary, and storage resolver.

        Args:
            store: Historical prototype transactional SQLite execution store.
            app_service: Synchronous provider boundary used by prototype resolution.
            storage: Prototype resolver authorizing provider locators against configured
                mounts.
            retry_seconds: Delay before retrying transient provider resolution.

        Raises:
            ValueError: A value fails the validation described above.
        """
        if retry_seconds <= 0:
            raise ValueError("resolution retry delay must be positive")
        self._store = store
        self._app_service = app_service
        self._storage = storage
        self._retry_seconds = retry_seconds

    def run_once(self, limit: int = 25) -> ResolutionResult:
        """Resolve a bounded set of due candidates and count outcomes without dispatching worker computation.

        Args:
            limit: Maximum number of eligible records to inspect or return.

        Returns:
            ResolutionResult: Counts of ready, deferred, and failed resolution
                candidates.
        """
        result = ResolutionResult()
        for candidate in self._store.list_resolution_candidates(limit):
            outcome = self._resolve_candidate(candidate)
            result = ResolutionResult(
                ready=result.ready + (outcome == "ready"),
                deferred=result.deferred + (outcome == "deferred"),
                failed=result.failed + (outcome == "failed"),
            )
        return result

    def _resolve_candidate(self, candidate: Execution) -> str:
        """Resolve configuration, plugin support, and authorized storage, then pin a ready specification or classify retry/terminal failure.

        Args:
            candidate: Due prototype execution selected for provider resolution.

        Returns:
            str: The identifier, path, or normalized text described above.
        """
        try:
            execution = self._store.begin_resolution(candidate.id)
        except InvalidTransitionError:
            return "ignored"
        try:
            if execution.failure_category is FailureCategory.APPLICATION:
                return self._settle_application_failure(execution)
            configuration = self._app_service.get_configuration(execution.empaia_job_id)
            direct_access = self._app_service.get_direct_access(
                execution.empaia_job_id, execution.input_name
            )
            if not direct_access.supported:
                raise AppServiceApplicationError(
                    "the WSI does not support configured direct access"
                )
            if (
                not execution.allowed_wsi_plugin_ids
                or direct_access.required_plugin_id
                not in execution.allowed_wsi_plugin_ids
            ):
                raise AppServiceApplicationError(
                    "the WSI representation is not enabled for this app"
                )
            try:
                locator = self._storage.resolve(direct_access.storage_path)
            except InvalidStorageInputError as error:
                raise AppServiceApplicationError(
                    "the WSI is unavailable through configured storage"
                ) from error
            except StorageUnavailableError:
                self._defer(execution)
                return "deferred"
            spec = ExecutionSpec(
                model_id=execution.model_id,
                model_version=execution.model_version,
                workflow=execution.workflow,
                resource_profile=execution.resource_profile,
                max_runtime_seconds=execution.max_runtime_seconds,
                wsi_id=direct_access.slide_id,
                storage_id=locator.storage_id,
                relative_locator=locator.relative_locator,
                output_name=execution.output_name,
                model_parameters=self._select_parameters(execution, configuration),
            )
            self._store.mark_ready(execution.id, spec)
        except AppServiceApplicationError:
            execution = self._store.record_resolution_application_failure(
                execution.id,
                "invalid_empaia_job",
                "the job input or configuration is not supported",
            )
            return self._settle_application_failure(execution)
        except AppServiceTransientError:
            self._defer(execution)
            return "deferred"
        except (AppServiceInfrastructureError, ValueError):
            self._store.fail_resolution_infrastructure(
                execution.id,
                "app_service_integration_error",
                "the EMPAIA application integration is unavailable",
            )
            return "failed"
        except InvalidTransitionError:
            # Cancellation or another control-plane action won the race.
            return "ignored"
        return "ready"

    @staticmethod
    def _select_parameters(
        execution: Execution, configuration: JobConfiguration
    ) -> dict[str, JsonValue]:
        """Select exactly admitted parameter names, rejecting missing values and ambiguous global/customer definitions.

        Args:
            execution: Current execution snapshot or public view to inspect or
                transition.
            configuration: Resolved workflow values or provider global/customer sections
                to validate.

        Returns:
            dict[str, JsonValue]: Decoded, serialized, or recorded fields described
                above.

        Raises:
            AppServiceApplicationError: The prototype provider rejects the external
                job's data or configuration.
        """
        ambiguous = (
            set(execution.model_parameter_names)
            & set(configuration.global_values)
            & set(configuration.customer_values)
        )
        if ambiguous:
            raise AppServiceApplicationError(
                "job configuration contains an ambiguous model parameter"
            )
        combined = configuration.global_values | configuration.customer_values
        missing = set(execution.model_parameter_names) - set(combined)
        if missing:
            raise AppServiceApplicationError(
                "job configuration is missing a required model parameter"
            )
        return {name: combined[name] for name in execution.model_parameter_names}

    def _settle_application_failure(self, execution: Execution) -> str:
        """Report a pending unsupported-job failure externally, retrying temporary reporting errors before durable failure settlement.

        Args:
            execution: Current execution snapshot or public view to inspect or
                transition.

        Returns:
            str: The identifier, path, or normalized text described above.
        """
        try:
            self._app_service.report_failure(
                execution.empaia_job_id,
                "The job input or configuration is not supported by this application.",
            )
        except AppServiceTransientError:
            self._defer(execution)
            return "deferred"
        except (AppServiceApplicationError, AppServiceInfrastructureError):
            self._store.fail_resolution_infrastructure(
                execution.id,
                "failure_reporting_error",
                "the EMPAIA application failure could not be reported",
            )
            return "failed"
        self._store.settle_resolution_application_failure(execution.id)
        return "failed"

    def _defer(self, execution: Execution) -> None:
        """Schedule a future resolution pass while tolerating a concurrent lifecycle change.

        Args:
            execution: Current execution snapshot or public view to inspect or
                transition.

        Returns:
            None: No return value.
        """
        retry_at = datetime.now(UTC) + timedelta(seconds=self._retry_seconds)
        with suppress(InvalidTransitionError):
            self._store.defer_resolution(execution.id, retry_at)

    async def run(self, stop: asyncio.Event, interval_seconds: float) -> None:
        """Run synchronous resolution in a thread, logging failed passes and polling until shutdown.

        Args:
            stop: Event signalled when the owning worker/resolver lifetime should end.
            interval_seconds: Positive polling interval between resolver or reconciler
                passes.

        Returns:
            None: No return value.

        Raises:
            ValueError: A value fails the validation described above.
        """
        if interval_seconds <= 0:
            raise ValueError("resolution interval must be positive")
        while not stop.is_set():
            try:
                await asyncio.to_thread(self.run_once)
            except Exception:
                logger.exception("unexpected resolution pass failure")
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
