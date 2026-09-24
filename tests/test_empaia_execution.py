from __future__ import annotations

import base64
import json
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from execution_scheduler.contracts import (
    InvalidScheduleRequestError,
    ScheduleRequest,
    SchedulingUnavailableError,
)
from execution_scheduler.dispatch import InMemoryWorkflowDispatch
from execution_scheduler.domain import RetryPolicy
from execution_scheduler.providers import ProviderFinishFailureError, ProviderSuccess
from execution_scheduler.repository import InMemoryExecutionRepository
from execution_scheduler.scheduler import ExecutionScheduler
from execution_scheduler.worker import ExecutionWorker
from integrations.empaia.gateway import EmpaiaGatewayError, EmpaiaJobContext
from integrations.empaia.input_access import EmpaiaInputAccessAdapter
from integrations.empaia.provider import EmpaiaProviderAdapter
from workflow_execution.application import WorkflowApplication
from workflow_execution.catalog import WorkflowCatalog
from workflow_execution.contracts import (
    SharedArtifact,
    WorkflowInvocation,
    WorkflowResult,
)


if TYPE_CHECKING:
    from integrations.empaia.gateway import MutationGuard
    from workflow_execution.application import WorkflowContext
    from workflow_execution.contracts import JsonValue, WorkflowOutput


class ControlledEmpaiaService:
    def __init__(self) -> None:
        self.configuration: dict[str, JsonValue] = {
            "output_path": "heatmap.tiff",
            "stride_fraction": 0.5,
            "output_tile_height": 256,
            "output_tile_width": 256,
        }
        self.configuration_reads = 0
        self.credentials: list[str] = []
        self.materialized: list[str] = []
        self.published: list[tuple[str, WorkflowOutput]] = []
        self.publish_failures: list[EmpaiaGatewayError] = []
        self.finalize_failures: list[EmpaiaGatewayError] = []
        self.close_failures: list[EmpaiaGatewayError] = []
        self.finalized = False
        self.failures: list[str] = []

    def gateway(self, context: EmpaiaJobContext) -> FakeGateway:
        self.credentials.append(context.credential)
        return FakeGateway(self, context.api_url, context.job_id)


class FakeGateway:
    def __init__(
        self,
        service: ControlledEmpaiaService,
        api_url: str,
        job_id: str,
    ) -> None:
        self._service = service
        self._api_url = api_url
        self._job_id = job_id
        self._before_mutation: MutationGuard | None = None

    def set_before_mutation(self, callback: MutationGuard) -> None:
        self._before_mutation = callback

    async def get_configuration(self) -> dict[str, JsonValue]:
        self._service.configuration_reads += 1
        return dict(self._service.configuration)

    async def materialize_input(self, name: str, destination: str) -> None:
        self._service.materialized.append(name)
        Path(destination).write_bytes(f"{name}-from-empaia".encode())

    async def publish_output(self, name: str, output: WorkflowOutput) -> None:
        if self._before_mutation is not None:
            await self._before_mutation()
        if self._service.publish_failures:
            raise self._service.publish_failures.pop(0)
        if isinstance(output, SharedArtifact) and not Path(output.path).is_file():
            raise AssertionError("durable workflow artifact disappeared")
        self._service.published.append((name, output))

    async def finalize(self) -> None:
        if self._before_mutation is not None:
            await self._before_mutation()
        if self._service.finalize_failures:
            raise self._service.finalize_failures.pop(0)
        self._service.finalized = True

    async def report_failure(self, message: str) -> None:
        if self._before_mutation is not None:
            await self._before_mutation()
        self._service.failures.append(message)

    async def close(self) -> None:
        if self._service.close_failures:
            raise self._service.close_failures.pop(0)


class CopyInputsWorkflow:
    async def run(self, context: WorkflowContext) -> WorkflowResult:
        slide = context.input_path("slide").read_bytes()
        roi = context.input_path("region_of_interest").read_bytes()
        output = context.output_path("heatmap.tiff")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(slide + b"|" + roi)
        return WorkflowResult(
            outputs={
                "heatmap": SharedArtifact(
                    path=str(output),
                    media_type="image/tiff",
                )
            }
        )


class EmpaiaExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_workflow_defaults_fill_missing_empaia_configuration(self) -> None:
        now = datetime.now(UTC)
        service = ControlledEmpaiaService()
        service.configuration = {}
        catalog = WorkflowCatalog.default()
        repository = InMemoryExecutionRepository(clock=lambda: now)
        provider = EmpaiaProviderAdapter(catalog, service.gateway)
        scheduler = ExecutionScheduler(
            repository=repository,
            providers={provider.name: provider},
            known_workflows=catalog.identifiers,
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=lambda: now,
        )

        scheduled = await scheduler.schedule(self._request(now, "credential"))
        execution = await repository.get(scheduled.execution.id)

        self.assertEqual(
            execution.invocation.resolved_configuration,
            {
                "output_path": "heatmap.tiff",
                "stride_fraction": 0.5,
                "output_tile_height": 256,
                "output_tile_width": 256,
            },
        )

    async def test_gateway_factory_failure_is_classified_during_prepare(self) -> None:
        def unavailable_gateway(context: EmpaiaJobContext) -> FakeGateway:
            del context
            raise EmpaiaGatewayError(
                "EMPAIA is temporarily unavailable",
                retryable=True,
            )

        now = datetime.now(UTC)
        catalog = WorkflowCatalog.default()
        provider = EmpaiaProviderAdapter(catalog, unavailable_gateway)
        scheduler = ExecutionScheduler(
            repository=InMemoryExecutionRepository(clock=lambda: now),
            providers={provider.name: provider},
            known_workflows=catalog.identifiers,
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=lambda: now,
        )

        with self.assertRaises(SchedulingUnavailableError):
            await scheduler.schedule(self._request(now, "credential"))

    async def test_gateway_factory_failure_is_classified_during_finish(self) -> None:
        def unavailable_gateway(context: EmpaiaJobContext) -> FakeGateway:
            del context
            raise EmpaiaGatewayError(
                "EMPAIA is temporarily unavailable",
                retryable=True,
            )

        provider = EmpaiaProviderAdapter(WorkflowCatalog.default(), unavailable_gateway)

        with self.assertRaises(ProviderFinishFailureError) as raised:
            await provider.finish(
                {
                    "api_url": "https://empaia.invalid",
                    "job_id": "job-1",
                    "credential": "credential",
                },
                ProviderSuccess(WorkflowResult(outputs={})),
            )

        self.assertTrue(raised.exception.retryable)
        self.assertFalse(raised.exception.outcome_unknown)

    async def test_gateway_cleanup_does_not_replace_successful_finish(self) -> None:
        service = ControlledEmpaiaService()
        service.close_failures.append(
            EmpaiaGatewayError(
                "EMPAIA cleanup failed",
                retryable=True,
            )
        )
        provider = EmpaiaProviderAdapter(WorkflowCatalog.default(), service.gateway)

        await provider.finish(
            {
                "api_url": "https://empaia.invalid",
                "job_id": "job-1",
                "credential": "credential",
            },
            ProviderSuccess(WorkflowResult(outputs={})),
        )

        self.assertTrue(service.finalized)

    async def test_controlled_empaia_lifecycle_and_admitted_replay(self) -> None:
        now = datetime.now(UTC)
        service = ControlledEmpaiaService()
        catalog = WorkflowCatalog.default()
        provider = EmpaiaProviderAdapter(catalog, service.gateway)
        repository = InMemoryExecutionRepository(clock=lambda: now)
        scheduler = ExecutionScheduler(
            repository=repository,
            providers={provider.name: provider},
            known_workflows=catalog.identifiers,
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=lambda: now,
        )
        with TemporaryDirectory() as directory:
            application = WorkflowApplication(
                catalog.get("prostate-heatmap"),
                CopyInputsWorkflow(),
                access_adapters={
                    "empaia": EmpaiaInputAccessAdapter(
                        service.gateway,
                        Path(directory),
                    )
                },
            )
            worker = ExecutionWorker(
                repository=repository,
                providers={provider.name: provider},
                dispatch=InMemoryWorkflowDispatch(
                    {"prostate-heatmap": application.run}
                ),
                owner="worker-1",
                lease_duration=timedelta(minutes=1),
            )
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
        now = datetime.now(UTC)
        expiry = now + timedelta(minutes=10)
        service = ControlledEmpaiaService()
        catalog = WorkflowCatalog.default()
        provider = EmpaiaProviderAdapter(catalog, service.gateway)
        repository = InMemoryExecutionRepository(clock=lambda: now)
        scheduler = ExecutionScheduler(
            repository=repository,
            providers={provider.name: provider},
            known_workflows=catalog.identifiers,
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=lambda: now,
        )

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
        now = datetime.now(UTC)
        service = ControlledEmpaiaService()
        service.configuration["stride_fraction"] = 0
        catalog = WorkflowCatalog.default()
        repository = InMemoryExecutionRepository(clock=lambda: now)
        provider = EmpaiaProviderAdapter(catalog, service.gateway)
        scheduler = ExecutionScheduler(
            repository=repository,
            providers={provider.name: provider},
            known_workflows=catalog.identifiers,
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=lambda: now,
        )

        request = self._request(now, "credential")
        with self.assertRaises(InvalidScheduleRequestError):
            await scheduler.schedule(request)

        self.assertEqual(service.configuration_reads, 1)
        self.assertIsNone(await repository.find_submission(provider.identify(request)))

    async def test_known_preacceptance_publication_failure_retries_result_only(
        self,
    ) -> None:
        now = datetime.now(UTC)
        service = ControlledEmpaiaService()
        service.publish_failures.append(
            EmpaiaGatewayError(
                "EMPAIA is temporarily unavailable",
                retryable=True,
            )
        )
        catalog = WorkflowCatalog.default()
        provider = EmpaiaProviderAdapter(catalog, service.gateway)
        repository = InMemoryExecutionRepository(
            clock=lambda: now,
            finish_policy=RetryPolicy(3, timedelta(0), timedelta(0)),
        )
        scheduler = ExecutionScheduler(
            repository=repository,
            providers={provider.name: provider},
            known_workflows=catalog.identifiers,
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=lambda: now,
        )
        workflow_runs = 0

        with TemporaryDirectory() as directory:
            application = WorkflowApplication(
                catalog.get("prostate-heatmap"),
                CopyInputsWorkflow(),
                access_adapters={
                    "empaia": EmpaiaInputAccessAdapter(
                        service.gateway,
                        Path(directory),
                    )
                },
            )

            async def run(invocation: WorkflowInvocation) -> WorkflowResult:
                nonlocal workflow_runs
                workflow_runs += 1
                return await application.run(invocation)

            worker = ExecutionWorker(
                repository=repository,
                providers={provider.name: provider},
                dispatch=InMemoryWorkflowDispatch({"prostate-heatmap": run}),
                owner="worker-1",
                lease_duration=timedelta(minutes=1),
            )
            admitted = await scheduler.schedule(self._request(now, "credential"))

            await worker.run_once()
            await worker.run_once()
            retrying = await scheduler.get(admitted.execution.id)
            await worker.run_once()
            completed = await scheduler.get(admitted.execution.id)

        self.assertEqual(retrying.status, "running")
        self.assertEqual(completed.status, "succeeded")
        self.assertEqual(workflow_runs, 1)
        self.assertEqual([name for name, _ in service.published], ["heatmap"])

    async def test_finalization_failure_after_upload_is_not_retried(self) -> None:
        now = datetime.now(UTC)
        service = ControlledEmpaiaService()
        service.finalize_failures.append(
            EmpaiaGatewayError(
                "EMPAIA finalization outcome is unknown",
                retryable=True,
            )
        )
        catalog = WorkflowCatalog.default()
        provider = EmpaiaProviderAdapter(catalog, service.gateway)
        repository = InMemoryExecutionRepository(
            clock=lambda: now,
            finish_policy=RetryPolicy(3, timedelta(0), timedelta(0)),
        )
        scheduler = ExecutionScheduler(
            repository=repository,
            providers={provider.name: provider},
            known_workflows=catalog.identifiers,
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=lambda: now,
        )
        with TemporaryDirectory() as directory:
            application = WorkflowApplication(
                catalog.get("prostate-heatmap"),
                CopyInputsWorkflow(),
                access_adapters={
                    "empaia": EmpaiaInputAccessAdapter(
                        service.gateway,
                        Path(directory),
                    )
                },
            )
            worker = ExecutionWorker(
                repository=repository,
                providers={provider.name: provider},
                dispatch=InMemoryWorkflowDispatch(
                    {"prostate-heatmap": application.run}
                ),
                owner="worker-1",
                lease_duration=timedelta(minutes=1),
            )
            admitted = await scheduler.schedule(self._request(now, "credential"))

            await worker.run_once()
            await worker.run_once()
            failed = await scheduler.get(admitted.execution.id)

        self.assertEqual(failed.status, "failed")
        self.assertEqual([name for name, _ in service.published], ["heatmap"])
        self.assertFalse(await worker.run_once())

    @staticmethod
    def _request(
        now: datetime,
        credential: str,
        *,
        expiry: datetime | None = None,
        job_id: str = "job-1",
    ) -> ScheduleRequest:
        return ScheduleRequest(
            workflow_id="prostate-heatmap",
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
    header = _base64_json({"alg": "test"})
    payload = _base64_json({"exp": expiry.timestamp(), "marker": marker})
    return f"{header}.{payload}.test-signature"


def _base64_json(value: dict[str, object]) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(value).encode()).decode()
    return encoded.rstrip("=")


if __name__ == "__main__":
    unittest.main()
