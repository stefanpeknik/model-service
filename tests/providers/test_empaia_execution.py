"""Check EMPAIA admission, publication retry boundaries, and credential-limited deadlines."""

from __future__ import annotations

import base64
import json
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from model_service.builder_deployment.runner import BuilderApplication
from model_service.builders.heatmap.spec import HeatmapOutputs
from model_service.builders.registry import get_builder
from model_service.contracts.invocation import ExecutionContext, Invocation
from model_service.contracts.model_declarations import ModelBinding, SpatialScoresModel
from model_service.contracts.result import BuilderResult, SharedArtifact
from model_service.providers.empaia.app_service.gateway import (
    EmpaiaGatewayError,
    EmpaiaInputState,
    EmpaiaJobContext,
    EmpaiaJobMode,
)
from model_service.providers.empaia.jobs.adapter import EmpaiaProviderAdapter
from model_service.providers.empaia.jobs.inputs import EmpaiaInputAccessFactory
from model_service.providers.empaia.publishing.publishers import (
    Commit,
    PublisherRegistry,
)
from model_service.scheduler.execution_scheduler import ExecutionScheduler
from model_service.scheduler.internal_state import RetryPolicy
from model_service.scheduler.public import (
    InvalidScheduleRequestError,
    Priority,
    ScheduleRequest,
    SchedulingUnavailableError,
)
from model_service.scheduler.running_work import ActiveWorkRegistry
from model_service.scheduler.worker import ExecutionWorker
from tests.support.builder_fixtures import deployments
from tests.support.memory_repository import InMemoryExecutionRepository


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from model_service.builders.base import BuilderContext
    from model_service.contracts.json_values import JsonValue
    from model_service.contracts.result import BuilderOutput
    from model_service.providers.empaia.app_service.gateway import (
        EmpaiaGateway,
        MutationGuard,
    )


MODEL = SpatialScoresModel(
    id="prostate-classifier-1",
    tile_size=512,
    output_tile_size=1,
    n_channels=1,
    mpp=0.468,
)
DEFAULT_CONFIGURATION: dict[str, JsonValue] = {
    "stride_fraction": 0.5,
    "skip_background": True,
}


API_URL = "https://empaia.invalid"


def resolve_model(builder_id: str, model_id: str) -> ModelBinding:
    if model_id != MODEL.id:
        raise KeyError(model_id)
    get_builder(builder_id).check_model(MODEL)
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
        {"empaia": EmpaiaInputAccessFactory(service.gateway, Path(directory))},
        model_handles=deployments(MODEL),
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
        self.mode = EmpaiaJobMode.STANDALONE
        self.finalized = False
        self.failures: list[str] = []

    def gateway(self, context: EmpaiaJobContext) -> FakeGateway:
        """Construct a job gateway backed by shared observable test state."""
        self.credentials.append(context.credential)
        return FakeGateway(self, context.api_url, context.job_id)


class RecordingPublisher:
    """Hand each heatmap to the fake gateway, which records it."""

    def accepts(self, model: ModelBinding) -> bool:
        return True

    async def prepare(
        self, gateway: EmpaiaGateway, name: str, output: BuilderOutput
    ) -> Commit:
        assert isinstance(gateway, FakeGateway)
        return lambda: gateway.publish_output(name, output)


PUBLISHERS = PublisherRegistry(
    {("continuous_pixelmap", "application/x-npz"): RecordingPublisher()}
)


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

    async def get_mode(self) -> EmpaiaJobMode:
        return self._service.mode

    async def has_input(self, name: str) -> bool:
        """Report the job's inputs; all of them unless a test removes one."""
        return name not in self._service.absent_inputs

    async def materialize_input(
        self,
        name: str,
        directory: str,
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
        slide = context.inputs.slide.read_bytes()
        roi = context.inputs.region_of_interest.read_bytes()
        output = context.output_path("heatmap")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(slide + b"|" + roi)
        return HeatmapOutputs(heatmap=SharedArtifact(path=str(output)))


class EmpaiaExecutionTests(unittest.IsolatedAsyncioTestCase):
    """Exercise EMPAIA admission, publication retry boundaries, and credential-limited deadlines."""

    async def test_builder_defaults_fill_missing_empaia_configuration(self) -> None:
        now = datetime.now(UTC)
        service = ControlledEmpaiaService()
        service.configuration = {}
        repository = InMemoryExecutionRepository(clock=lambda: now)
        provider = EmpaiaProviderAdapter(
            service.gateway, allowed_api_urls=(API_URL,), publishers=PUBLISHERS
        )
        scheduler = scheduler_for(repository, provider, now)

        scheduled = await scheduler.schedule(self._request(now, "credential"))
        execution = await repository.get(scheduled.execution.id)

        self.assertEqual(
            execution.invocation.resolved_configuration, DEFAULT_CONFIGURATION
        )
        self.assertEqual(execution.builder_id, "heatmap")
        self.assertEqual(execution.model_id, MODEL.id)

    async def test_configuration_keys_the_builder_does_not_take_are_ignored(
        self,
    ) -> None:
        now = datetime.now(UTC)
        service = ControlledEmpaiaService()
        service.configuration["viewer_colormap"] = "viridis"
        repository = InMemoryExecutionRepository(clock=lambda: now)
        provider = EmpaiaProviderAdapter(
            service.gateway, allowed_api_urls=(API_URL,), publishers=PUBLISHERS
        )
        scheduler = scheduler_for(repository, provider, now)

        with self.assertLogs("model_service.providers.empaia", "WARNING") as logs:
            scheduled = await scheduler.schedule(self._request(now, "credential"))
        execution = await repository.get(scheduled.execution.id)

        self.assertEqual(
            execution.invocation.resolved_configuration, DEFAULT_CONFIGURATION
        )
        self.assertIn("viewer_colormap", logs.output[0])

    async def test_optional_inputs_the_job_lacks_are_left_out(self) -> None:
        now = datetime.now(UTC)
        service = ControlledEmpaiaService()
        service.absent_inputs = {"region_of_interest"}
        repository = InMemoryExecutionRepository(clock=lambda: now)
        provider = EmpaiaProviderAdapter(
            service.gateway, allowed_api_urls=(API_URL,), publishers=PUBLISHERS
        )
        scheduler = scheduler_for(repository, provider, now)

        scheduled = await scheduler.schedule(self._request(now, "credential"))
        execution = await repository.get(scheduled.execution.id)

        self.assertEqual(list(execution.invocation.inputs), ["slide"])

    async def test_the_jobs_mode_is_recorded(self) -> None:
        now = datetime.now(UTC)
        service = ControlledEmpaiaService()
        service.mode = EmpaiaJobMode.PREPROCESSING
        repository = InMemoryExecutionRepository(clock=lambda: now)
        provider = EmpaiaProviderAdapter(
            service.gateway, allowed_api_urls=(API_URL,), publishers=PUBLISHERS
        )
        scheduler = scheduler_for(repository, provider, now)

        scheduled = await scheduler.schedule(self._request(now, "credential"))
        execution = await repository.get(scheduled.execution.id)

        state = EmpaiaInputState.model_validate(execution.invocation.input_access.state)
        self.assertEqual(state.mode, EmpaiaJobMode.PREPROCESSING)
        self.assertEqual(execution.priority, Priority.NORMAL)

    async def test_a_job_started_by_hand_has_high_priority(self) -> None:
        now = datetime.now(UTC)
        service = ControlledEmpaiaService()
        repository = InMemoryExecutionRepository(clock=lambda: now)
        provider = EmpaiaProviderAdapter(
            service.gateway, allowed_api_urls=(API_URL,), publishers=PUBLISHERS
        )
        scheduler = scheduler_for(repository, provider, now)

        scheduled = await scheduler.schedule(self._request(now, "credential"))
        execution = await repository.get(scheduled.execution.id)

        self.assertIs(service.mode, EmpaiaJobMode.STANDALONE)
        self.assertEqual(execution.priority, Priority.HIGH)

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

    async def test_controlled_empaia_lifecycle_and_admitted_replay(self) -> None:
        """Verify controlled EMPAIA lifecycle and admitted replay."""
        now = datetime.now(UTC)
        service = ControlledEmpaiaService()
        provider = EmpaiaProviderAdapter(
            service.gateway, allowed_api_urls=(API_URL,), publishers=PUBLISHERS
        )
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
        # The slide is told which inputs may limit what it fetches.
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
        provider = EmpaiaProviderAdapter(
            service.gateway, allowed_api_urls=(API_URL,), publishers=PUBLISHERS
        )
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
        provider = EmpaiaProviderAdapter(
            service.gateway, allowed_api_urls=(API_URL,), publishers=PUBLISHERS
        )
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
        provider = EmpaiaProviderAdapter(
            service.gateway, allowed_api_urls=(API_URL,), publishers=PUBLISHERS
        )
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
        provider = EmpaiaProviderAdapter(
            service.gateway, allowed_api_urls=(API_URL,), publishers=PUBLISHERS
        )
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


class EmpaiaCredentialTests(unittest.IsolatedAsyncioTestCase):
    """Validate token structure and deadline caps without authenticating JWT signatures."""

    def setUp(self) -> None:
        """Create a provider with a controlled gateway and a future execution deadline."""
        self.service = ControlledEmpaiaService()
        self.provider = EmpaiaProviderAdapter(
            self.service.gateway, allowed_api_urls=(API_URL,)
        )
        self.deadline = datetime.now(UTC) + timedelta(hours=1)

    async def test_integer_and_fractional_expiry_preserve_utc_deadline_caps(
        self,
    ) -> None:
        """Keep NumericDate seconds and fractional seconds in the prepared execution."""
        for expiry in (
            int(self.deadline.timestamp()) - 300,
            self.deadline.timestamp() - 300.125,
        ):
            with self.subTest(expiry=expiry):
                prepared = await self.provider.prepare(
                    self._request(_token({"exp": expiry})), self.deadline
                )
                expected = datetime.fromtimestamp(expiry, tz=UTC)
                self.assertEqual(prepared.deadline_cap, expected)
                self.assertEqual(prepared.terminal_report_deadline_at, expected)
                self.assertEqual(prepared.invocation.deadline_at, expected)

    async def test_later_expiry_does_not_extend_the_execution_deadline(self) -> None:
        """Use the earlier execution deadline even when the credential lasts longer."""
        expiry = self.deadline + timedelta(hours=1)
        prepared = await self.provider.prepare(
            self._request(_token({"exp": expiry.timestamp()})), self.deadline
        )
        self.assertEqual(prepared.deadline_cap, expiry)
        self.assertEqual(prepared.invocation.deadline_at, self.deadline)

    async def test_invalid_expiry_is_rejected_before_gateway_preparation(self) -> None:
        """Reject coerced, nonfinite, and unrepresentable expiry with a safe error."""
        for expiry in (
            None,
            True,
            False,
            "123",
            [],
            {},
            float("nan"),
            float("inf"),
            float("-inf"),
            1e300,
            -1e300,
            1e20,
            10**400,
        ):
            with self.subTest(expiry=expiry):
                await self._assert_invalid(_token({"exp": expiry}))
        self.assertEqual(self.service.configuration_reads, 0)
        self.assertEqual(self.service.credentials, [])

    async def test_missing_claim_and_malformed_token_are_safe_admission_errors(
        self,
    ) -> None:
        """Translate JWT decoding errors without exposing the original credential."""
        for token in (
            "not-a-token",
            "a.b.c",
            "e30.!!!.",
            _token({}),
            _token([]),
            _token({"exp": 123}, header=[]),
        ):
            with self.subTest(token_shape=token.count(".")):
                await self._assert_invalid(token)
        self.assertEqual(self.service.configuration_reads, 0)

    async def test_nested_malformed_payload_raises_a_safe_error(self) -> None:
        """Cover the pre-verification payload parser regression fixed in PyJWT 2.15."""
        payload = (
            base64.urlsafe_b64encode(b"[" * 2000 + b"]" * 2000).rstrip(b"=").decode()
        )
        await self._assert_invalid(f"e30.{payload}.")

    async def _assert_invalid(self, token: str) -> None:
        """Assert the provider returns a safe validation error for the supplied token."""
        with self.assertRaises(InvalidScheduleRequestError) as raised:
            await self.provider.prepare(self._request(token), self.deadline)
        self.assertEqual(
            str(raised.exception), "EMPAIA credential does not contain a valid expiry"
        )
        self.assertNotIn(token, str(raised.exception))

    @staticmethod
    def _request(token: str) -> ScheduleRequest:
        """Build a synthetic job request without contacting an external AppService."""
        return ScheduleRequest(
            builder_id="heatmap",
            model_id="prostate-classifier-1",
            provider="empaia",
            body={
                "installation_id": "test-installation",
                "api_url": "https://empaia.invalid",
                "job_id": "test-job",
                "credential": token,
            },
        )


def _token(payload: object, *, header: object | None = None) -> str:
    """Encode synthetic token sections with an empty unverified signature."""

    def encode(value: object) -> str:
        """Encode fixture JSON using JWT's unpadded Base64url representation."""
        return (
            base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()
        )

    return f"{encode({} if header is None else header)}.{encode(payload)}."


if __name__ == "__main__":
    unittest.main()
