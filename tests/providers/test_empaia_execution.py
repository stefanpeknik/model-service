"""Check EMPAIA admission, publication retry boundaries, and credential-limited deadlines."""

from __future__ import annotations

import base64
import json
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from model_service.builders.application import BuilderApplication, validate_model
from model_service.builders.contracts import (
    BuilderResult,
    ExecutionContext,
    Invocation,
    ModelBinding,
    SharedArtifact,
    SpatialScoresModel,
)
from model_service.builders.registry import get_builder
from model_service.providers.empaia.gateway import EmpaiaGatewayError, EmpaiaJobContext
from model_service.providers.empaia.input_access import EmpaiaInputAccessAdapter
from model_service.providers.empaia.provider import EmpaiaProviderAdapter
from model_service.scheduler.active_work import ActiveWorkRegistry
from model_service.scheduler.contracts import (
    InvalidScheduleRequestError,
    ScheduleRequest,
    SchedulingUnavailableError,
)
from model_service.scheduler.domain import RetryPolicy
from model_service.scheduler.providers import (
    ProviderFinishFailureError,
    ProviderSuccess,
)
from model_service.scheduler.scheduler import ExecutionScheduler
from model_service.scheduler.worker import ExecutionWorker
from tests.support.memory_repository import InMemoryExecutionRepository


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from model_service.builders.application import BuilderContext
    from model_service.builders.contracts import BuilderOutput, JsonValue
    from model_service.providers.empaia.gateway import MutationGuard


MODEL = SpatialScoresModel(
    id="prostate-classifier-1",
    tile_size=512,
    output_tile_size=1,
    n_channels=1,
    mpp=0.468,
)
DEFAULT_CONFIGURATION: dict[str, JsonValue] = {
    "stride_fraction": 0.5,
    "output_tile_height": 256,
    "output_tile_width": 256,
    "encoding": "uint8",
    "skip_background": True,
}


API_URL = "https://empaia.invalid"


def resolve_model(builder_id: str, model_id: str) -> ModelBinding:
    if model_id != MODEL.id:
        raise KeyError(model_id)
    validate_model(get_builder(builder_id), MODEL)
    return MODEL


type Handler = Callable[
    [Invocation, ModelBinding, ExecutionContext], Awaitable[BuilderResult]
]


class Dispatch:
    """Route builder runs to in-process handlers instead of Ray."""

    def __init__(self, handlers: dict[str, Handler] | None = None) -> None:
        self._handlers = handlers or {}

    async def run(
        self,
        builder_id: str,
        invocation: Invocation,
        model: ModelBinding,
        execution_context: ExecutionContext,
    ) -> BuilderResult:
        return await self._handlers[builder_id](invocation, model, execution_context)


def scheduler_for(
    repository: InMemoryExecutionRepository,
    provider: EmpaiaProviderAdapter,
    now: datetime,
) -> ExecutionScheduler:
    return ExecutionScheduler(
        repository=repository,
        dispatch=Dispatch(),
        providers={provider.name: provider},
        resolve_model=resolve_model,
        default_timeout=timedelta(hours=1),
        maximum_timeout=timedelta(hours=6),
        clock=lambda: now,
    )


def worker_for(
    repository: InMemoryExecutionRepository,
    provider: EmpaiaProviderAdapter,
    run: Handler,
) -> ExecutionWorker:
    return ExecutionWorker(
        repository=repository,
        active_work=ActiveWorkRegistry(),
        providers={provider.name: provider},
        dispatch=Dispatch({"heatmap": run}),
        resolve_model=resolve_model,
        owner="worker-1",
        lease_duration=timedelta(minutes=1),
    )


def copy_inputs_application(
    service: ControlledEmpaiaService, directory: str
) -> BuilderApplication:
    return BuilderApplication(
        get_builder("heatmap"),
        CopyInputsBuilder(),
        {"empaia": EmpaiaInputAccessAdapter(service.gateway, Path(directory))},
    )


class ControlledEmpaiaService:
    """Keep observable configuration, publication state, and injected failures for fake job gateways."""

    def __init__(self) -> None:
        """Initialize valid configuration, observable call lists, and independently injectable publication/finalization/cleanup failures."""
        self.configuration: dict[str, JsonValue] = dict(DEFAULT_CONFIGURATION)
        self.configuration_reads = 0
        self.credentials: list[str] = []
        self.materialized: list[str] = []
        self.published: list[tuple[str, BuilderOutput]] = []
        self.publish_failures: list[EmpaiaGatewayError] = []
        self.finalize_failures: list[EmpaiaGatewayError] = []
        self.close_failures: list[EmpaiaGatewayError] = []
        self.absent_inputs: set[str] = set()
        self.finalized = False
        self.failures: list[str] = []

    def gateway(self, context: EmpaiaJobContext) -> FakeGateway:
        """Construct a job gateway backed by shared observable test state."""
        self.credentials.append(context.credential)
        return FakeGateway(self, context.api_url, context.job_id)


class FakeGateway:
    """Simulate one EMPAIA job while recording acquisition, mutation guards, publication, and cleanup."""

    def __init__(
        self,
        service: ControlledEmpaiaService,
        api_url: str,
        job_id: str,
    ) -> None:
        """Retain fake service/job identity and start without an external-mutation guard."""
        self._service = service
        self._api_url = api_url
        self._job_id = job_id
        self._before_mutation: MutationGuard | None = None

    def set_before_mutation(self, callback: MutationGuard) -> None:
        """Store the durable mutation guard for later fake external writes."""
        self._before_mutation = callback

    async def get_configuration(self) -> dict[str, JsonValue]:
        """Count configuration reads and return a copy of the configured values."""
        self._service.configuration_reads += 1
        return dict(self._service.configuration)

    async def has_input(self, name: str) -> bool:
        """Report the job's inputs; all of them unless a test removes one."""
        return name not in self._service.absent_inputs

    async def materialize_input(
        self, name: str, directory: str, *, mpp: float | None = None
    ) -> str:
        """Record the input name and write identifiable fixture bytes into the directory."""
        self._service.materialized.append(name)
        path = Path(directory) / f"{name}.bin"
        path.write_bytes(f"{name}-from-empaia".encode())
        return str(path)

    async def publish_output(self, name: str, output: BuilderOutput) -> None:
        """Await the mutation guard, inject any queued failure, and record a still-existing artifact or inline output."""
        if self._before_mutation is not None:
            await self._before_mutation()
        if self._service.publish_failures:
            raise self._service.publish_failures.pop(0)
        if isinstance(output, SharedArtifact) and not Path(output.path).is_file():
            raise AssertionError("durable builder artifact disappeared")
        self._service.published.append((name, output))

    async def finalize(self) -> None:
        """Await the guard, inject any queued finalization error, or record successful job finalization."""
        if self._before_mutation is not None:
            await self._before_mutation()
        if self._service.finalize_failures:
            raise self._service.finalize_failures.pop(0)
        self._service.finalized = True

    async def report_failure(self, message: str) -> None:
        """Await the mutation guard and record the reported safe terminal message."""
        if self._before_mutation is not None:
            await self._before_mutation()
        self._service.failures.append(message)

    async def close(self) -> None:
        """Raise the next queued cleanup failure, otherwise complete without resources to release."""
        if self._service.close_failures:
            raise self._service.close_failures.pop(0)


class CopyInputsBuilder:
    """Write both materialized inputs into the artifact, proving access reaches the builder."""

    async def run(self, context: BuilderContext) -> BuilderResult:
        slide = context.input_path("slide").read_bytes()
        roi = context.input_path("region_of_interest").read_bytes()
        output = context.output_path("heatmap")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(slide + b"|" + roi)
        return BuilderResult(
            outputs={
                "heatmap": SharedArtifact(
                    path=str(output),
                    media_type="image/tiff",
                )
            }
        )


class EmpaiaExecutionTests(unittest.IsolatedAsyncioTestCase):
    """Exercise EMPAIA admission, publication retry boundaries, and credential-limited deadlines."""

    async def test_builder_defaults_fill_missing_empaia_configuration(self) -> None:
        now = datetime.now(UTC)
        service = ControlledEmpaiaService()
        service.configuration = {}
        repository = InMemoryExecutionRepository(clock=lambda: now)
        provider = EmpaiaProviderAdapter(service.gateway, allowed_api_urls=(API_URL,))
        scheduler = scheduler_for(repository, provider, now)

        scheduled = await scheduler.schedule(self._request(now, "credential"))
        execution = await repository.get(scheduled.execution.id)

        self.assertEqual(
            execution.invocation.resolved_configuration, DEFAULT_CONFIGURATION
        )
        self.assertEqual(execution.builder_id, "heatmap")
        self.assertEqual(execution.model_id, MODEL.id)

    async def test_optional_inputs_the_job_lacks_are_left_out(self) -> None:
        now = datetime.now(UTC)
        service = ControlledEmpaiaService()
        service.absent_inputs = {"region_of_interest"}
        repository = InMemoryExecutionRepository(clock=lambda: now)
        provider = EmpaiaProviderAdapter(service.gateway, allowed_api_urls=(API_URL,))
        scheduler = scheduler_for(repository, provider, now)

        scheduled = await scheduler.schedule(self._request(now, "credential"))
        execution = await repository.get(scheduled.execution.id)

        self.assertEqual(list(execution.invocation.inputs), ["slide"])

    async def test_gateway_factory_failure_is_classified_during_prepare(self) -> None:
        """Verify gateway factory failure is classified during prepare."""

        def unavailable_gateway(context: EmpaiaJobContext) -> FakeGateway:
            """Inject classified retryable gateway-construction failure without opening a connection."""
            del context
            raise EmpaiaGatewayError(
                "EMPAIA is temporarily unavailable",
                retryable=True,
            )

        now = datetime.now(UTC)
        provider = EmpaiaProviderAdapter(
            unavailable_gateway, allowed_api_urls=(API_URL,)
        )
        scheduler = scheduler_for(
            InMemoryExecutionRepository(clock=lambda: now), provider, now
        )

        with self.assertRaises(SchedulingUnavailableError):
            await scheduler.schedule(self._request(now, "credential"))

    async def test_gateway_factory_failure_is_classified_during_finish(self) -> None:
        """Verify gateway factory failure is classified during finish."""

        def unavailable_gateway(context: EmpaiaJobContext) -> FakeGateway:
            """Inject classified retryable gateway-construction failure without opening a connection."""
            del context
            raise EmpaiaGatewayError(
                "EMPAIA is temporarily unavailable",
                retryable=True,
            )

        provider = EmpaiaProviderAdapter(
            unavailable_gateway, allowed_api_urls=(API_URL,)
        )

        with self.assertRaises(ProviderFinishFailureError) as raised:
            await provider.finish(
                {
                    "api_url": "https://empaia.invalid",
                    "job_id": "job-1",
                    "credential": "credential",
                },
                ProviderSuccess(BuilderResult(outputs={})),
            )

        self.assertTrue(raised.exception.retryable)
        self.assertFalse(raised.exception.outcome_unknown)

    async def test_gateway_cleanup_does_not_replace_successful_finish(self) -> None:
        """Verify gateway cleanup does not replace successful finish."""
        service = ControlledEmpaiaService()
        service.close_failures.append(
            EmpaiaGatewayError(
                "EMPAIA cleanup failed",
                retryable=True,
            )
        )
        provider = EmpaiaProviderAdapter(service.gateway, allowed_api_urls=(API_URL,))

        await provider.finish(
            {
                "api_url": "https://empaia.invalid",
                "job_id": "job-1",
                "credential": "credential",
            },
            ProviderSuccess(BuilderResult(outputs={})),
        )

        self.assertTrue(service.finalized)

    async def test_controlled_empaia_lifecycle_and_admitted_replay(self) -> None:
        """Verify controlled EMPAIA lifecycle and admitted replay."""
        now = datetime.now(UTC)
        service = ControlledEmpaiaService()
        provider = EmpaiaProviderAdapter(service.gateway, allowed_api_urls=(API_URL,))
        repository = InMemoryExecutionRepository(clock=lambda: now)
        scheduler = scheduler_for(repository, provider, now)
        with TemporaryDirectory() as directory:
            application = copy_inputs_application(service, directory)
            worker = worker_for(repository, provider, application.run)
            request = self._request(now, "original")

            admitted = await scheduler.schedule(request)
            replay = await scheduler.schedule(self._request(now, "replacement"))

            self.assertTrue(admitted.created)
            self.assertFalse(replay.created)
            self.assertEqual(replay.execution.id, admitted.execution.id)
            self.assertEqual(service.configuration_reads, 1)
            durable = await repository.get(admitted.execution.id)
            self.assertNotIn("credential", durable.invocation.input_access.state)

            await worker.run_once()
            processing_complete = await scheduler.get(admitted.execution.id)
            self.assertEqual(processing_complete.status, "running")
            self.assertIsNone(processing_complete.result)
            await worker.run_once()
            completed = await scheduler.get(admitted.execution.id)

        self.assertEqual(completed.status, "succeeded")
        self.assertEqual(service.materialized, ["slide", "region_of_interest"])
        self.assertEqual([name for name, _ in service.published], ["heatmap"])
        self.assertTrue(service.finalized)
        self.assertEqual(
            service.credentials,
            [
                _credential(now + timedelta(hours=2), "original"),
                _credential(now + timedelta(hours=2), "original"),
                _credential(now + timedelta(hours=2), "original"),
            ],
        )

    async def test_credential_expiry_caps_the_execution_deadline(self) -> None:
        """Verify credential expiry caps the execution deadline."""
        now = datetime.now(UTC)
        expiry = now + timedelta(minutes=10)
        service = ControlledEmpaiaService()
        provider = EmpaiaProviderAdapter(service.gateway, allowed_api_urls=(API_URL,))
        repository = InMemoryExecutionRepository(clock=lambda: now)
        scheduler = scheduler_for(repository, provider, now)

        admitted = await scheduler.schedule(
            self._request(now, "credential", expiry=expiry)
        )
        durable = await repository.get(admitted.execution.id)

        self.assertEqual(durable.deadline_at, expiry)
        self.assertEqual(durable.invocation.deadline_at, expiry)

        with self.assertRaises(InvalidScheduleRequestError):
            await scheduler.schedule(
                self._request(
                    now,
                    "expired-credential",
                    expiry=now - timedelta(seconds=1),
                    job_id="expired-job",
                )
            )

    async def test_invalid_configuration_is_rejected_before_admission(self) -> None:
        """Verify invalid configuration is rejected before admission."""
        now = datetime.now(UTC)
        service = ControlledEmpaiaService()
        service.configuration["stride_fraction"] = 0
        repository = InMemoryExecutionRepository(clock=lambda: now)
        provider = EmpaiaProviderAdapter(service.gateway, allowed_api_urls=(API_URL,))
        scheduler = scheduler_for(repository, provider, now)

        request = self._request(now, "credential")
        with self.assertRaises(InvalidScheduleRequestError):
            await scheduler.schedule(request)

        self.assertEqual(service.configuration_reads, 1)
        self.assertIsNone(await repository.find_submission(provider.identify(request)))

    async def test_known_preacceptance_publication_failure_retries_result_only(
        self,
    ) -> None:
        """Verify known preacceptance publication failure retries result only."""
        now = datetime.now(UTC)
        service = ControlledEmpaiaService()
        service.publish_failures.append(
            EmpaiaGatewayError(
                "EMPAIA is temporarily unavailable",
                retryable=True,
            )
        )
        provider = EmpaiaProviderAdapter(service.gateway, allowed_api_urls=(API_URL,))
        repository = InMemoryExecutionRepository(
            clock=lambda: now,
            finish_policy=RetryPolicy(3, timedelta(0), timedelta(0)),
        )
        scheduler = scheduler_for(repository, provider, now)
        builder_runs = 0

        with TemporaryDirectory() as directory:
            application = copy_inputs_application(service, directory)

            async def run(
                invocation: Invocation,
                model: ModelBinding,
                execution_context: ExecutionContext,
            ) -> BuilderResult:
                nonlocal builder_runs
                builder_runs += 1
                return await application.run(invocation, model, execution_context)

            worker = worker_for(repository, provider, run)
            admitted = await scheduler.schedule(self._request(now, "credential"))

            await worker.run_once()
            await worker.run_once()
            retrying = await scheduler.get(admitted.execution.id)
            await worker.run_once()
            completed = await scheduler.get(admitted.execution.id)

        self.assertEqual(retrying.status, "running")
        self.assertEqual(completed.status, "succeeded")
        self.assertEqual(builder_runs, 1)
        self.assertEqual([name for name, _ in service.published], ["heatmap"])

    async def test_finalization_failure_after_upload_reports_failure_instead_of_retrying(
        self,
    ) -> None:
        """Verify finalization failure after upload is not retried."""
        now = datetime.now(UTC)
        service = ControlledEmpaiaService()
        service.finalize_failures.append(
            EmpaiaGatewayError(
                "EMPAIA finalization outcome is unknown",
                retryable=True,
            )
        )
        provider = EmpaiaProviderAdapter(service.gateway, allowed_api_urls=(API_URL,))
        repository = InMemoryExecutionRepository(
            clock=lambda: now,
            finish_policy=RetryPolicy(3, timedelta(0), timedelta(0)),
        )
        scheduler = scheduler_for(repository, provider, now)
        with TemporaryDirectory() as directory:
            application = copy_inputs_application(service, directory)
            worker = worker_for(repository, provider, application.run)
            admitted = await scheduler.schedule(self._request(now, "credential"))

            await worker.run_once()
            await worker.run_once()
            failed = await scheduler.get(admitted.execution.id)

            # The publication is not retried, but EMPAIA is told the job failed.
            self.assertTrue(await worker.run_once())

        self.assertEqual(failed.status, "failed")
        self.assertEqual([name for name, _ in service.published], ["heatmap"])
        self.assertFalse(service.finalized)
        self.assertEqual(len(service.failures), 1)
        self.assertFalse(await worker.run_once())

    @staticmethod
    def _request(
        now: datetime,
        credential: str,
        *,
        expiry: datetime | None = None,
        job_id: str = "job-1",
    ) -> ScheduleRequest:
        """Build a provider request for the selected fixture identity and deadline settings."""
        return ScheduleRequest(
            builder_id="heatmap",
            model_id=MODEL.id,
            provider="empaia",
            body={
                "installation_id": "installation-1",
                "api_url": "https://empaia.invalid",
                "job_id": job_id,
                "credential": _credential(
                    expiry or now + timedelta(hours=2),
                    credential,
                ),
            },
            requested_timeout_seconds=3600,
        )


def _credential(expiry: datetime, marker: str) -> str:
    """Build an unsigned synthetic JWT-shaped credential with an expiry claim for validation fixtures."""
    header = _base64_json({"alg": "test"})
    payload = _base64_json({"exp": expiry.timestamp(), "marker": marker})
    signature = base64.urlsafe_b64encode(b"test-signature").rstrip(b"=").decode()
    return f"{header}.{payload}.{signature}"


def _base64_json(value: dict[str, object]) -> str:
    """Encode JSON as unpadded URL-safe base64 for a synthetic test credential."""
    encoded = base64.urlsafe_b64encode(json.dumps(value).encode()).decode()
    return encoded.rstrip("=")


if __name__ == "__main__":
    unittest.main()
