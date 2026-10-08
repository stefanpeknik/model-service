"""Check assembled EMPAIA execution through real PostgreSQL and local ASGI HTTP boundaries.

Requires TEST_POSTGRES_DSN and creates an isolated schema; provider communication is
controlled locally through ASGI transports.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import unittest
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from uuid import uuid4

import httpx
from fastapi import FastAPI
from psycopg import AsyncConnection, sql

from execution_scheduler.active_work import ActiveWorkRegistry
from execution_scheduler.contracts import ScheduleRequest
from execution_scheduler.domain import RetryPolicy, WorkKind
from execution_scheduler.persistence.postgres import PostgresExecutionRepository
from execution_scheduler.persistence.protected_context import ProtectedContextCipher
from execution_scheduler.scheduler import ExecutionScheduler
from execution_scheduler.worker import ExecutionWorker
from integrations.empaia.input_access import EmpaiaInputAccessAdapter
from integrations.empaia.provider import EmpaiaProviderAdapter
from model_service.api import create_execution_app
from workflow_execution.application import WorkflowApplication, WorkflowContext
from workflow_execution.catalog import (
    WorkflowCatalog,
    WorkflowConfiguration,
    WorkflowDefinition,
    WorkflowInputDefinition,
    WorkflowValueDefinition,
    validate_empty_configuration,
)
from workflow_execution.contracts import (
    JsonValue,
    SharedArtifact,
    WorkflowOutput,
    WorkflowResult,
)
from workflow_execution.dispatch import HTTPWorkflowDispatch, InMemoryWorkflowDispatch
from workflow_execution.http import (
    WorkflowRunRequest,  # noqa: TC001 - FastAPI resolves endpoint annotations at runtime.
)


if TYPE_CHECKING:
    from integrations.empaia.gateway import EmpaiaJobContext, MutationGuard


@dataclass(slots=True)
class ControlledEmpaiaService:
    """Keep observable configuration, publication state, and injected failures for fake job gateways."""

    source: Path
    published: dict[str, WorkflowOutput] = field(default_factory=dict)
    finalized: bool = False

    def gateway(self, context: EmpaiaJobContext) -> ControlledEmpaiaGateway:
        """Construct a job gateway backed by shared observable test state.

        Args:
            context: External job origin/ID and protected bearer credential.

        Returns:
            ControlledEmpaiaGateway: A local controlled gateway without external HTTP
                communication.
        """
        del context
        return ControlledEmpaiaGateway(self)


class ControlledEmpaiaGateway:
    """Materialize local fixture inputs and record guarded publication without real AppService HTTP."""

    def __init__(self, service: ControlledEmpaiaService) -> None:
        """Retain the shared service state and start without a mutation guard.

        Args:
            service: Shared controlled provider state used by a fake gateway.
        """
        self._service = service
        self._before_mutation: MutationGuard | None = None

    def set_before_mutation(self, callback: MutationGuard) -> None:
        """Store the callback required before fake external mutations.

        Args:
            callback: Async callback invoked before an external provider mutation.

        Returns:
            None: No return value.
        """
        self._before_mutation = callback

    async def get_configuration(self) -> dict[str, JsonValue]:
        """Return empty provider overrides so catalog defaults resolve the invocation.

        Returns:
            dict[str, JsonValue]: Decoded, serialized, or recorded fields described
                above.
        """
        return {}

    async def materialize_input(self, _name: str, destination: str) -> None:
        """Copy the service fixture file to the requested materialization destination.

        Args:
            _name: Unused input/output name or configuration field selected by this
                operation; retained to match the interface.
            destination: Local path where input materialization or mask writing must
                produce a file.

        Returns:
            None: No return value.
        """
        shutil.copyfile(self._service.source, destination)

    async def publish_output(self, name: str, output: WorkflowOutput) -> None:
        """Await the mutation guard and store the named result in shared service state.

        Args:
            name: Input/output name or configuration field selected by this operation.
            output: Inline JSON value or completed shared artifact to publish.

        Returns:
            None: No return value.
        """
        if self._before_mutation is not None:
            await self._before_mutation()
        self._service.published[name] = output

    async def finalize(self) -> None:
        """Await the mutation guard and mark the fake job finalized.

        Returns:
            None: No return value.
        """
        if self._before_mutation is not None:
            await self._before_mutation()
        self._service.finalized = True

    async def report_failure(self, message: str) -> None:
        """Fail if this success-path fixture unexpectedly receives a terminal failure report.

        Args:
            message: Safe user-facing failure or cancellation text.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        raise AssertionError(message)

    async def close(self) -> None:
        """Complete without cleanup because this gateway owns no network resources.

        Returns:
            None: No return value.
        """
        return None


class CopyWorkflow:
    """Copy a materialized provider input into a shared result for assembled persistence/HTTP tests."""

    async def run(self, context: WorkflowContext) -> WorkflowResult:
        """Copy the source input into a result file and return its binary artifact reference.

        Args:
            context: Invocation-scoped input/output access, resolved configuration, and
                absolute deadline.

        Returns:
            WorkflowResult: Named inline outputs and/or references to completed artifact
                files.
        """
        output = context.output_path("result.bin")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(context.input_path("source"), output)
        return WorkflowResult(
            outputs={
                "result": SharedArtifact(
                    path=str(output),
                    media_type="application/octet-stream",
                )
            }
        )


@unittest.skipUnless(
    os.environ.get("TEST_POSTGRES_DSN"),
    "TEST_POSTGRES_DSN is not configured",
)
class DurableEmpaiaExecutionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Exercise assembled EMPAIA execution through real PostgreSQL and local ASGI HTTP boundaries."""

    async def test_public_empaia_lifecycle_crosses_real_persistence_and_http(
        self,
    ) -> None:
        """Verify public EMPAIA lifecycle crosses real persistence and HTTP.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        dsn = os.environ["TEST_POSTGRES_DSN"]
        schema = f"test_empaia_assembled_{uuid4().hex}"
        policy = RetryPolicy(3, timedelta(0), timedelta(0))
        repository = PostgresExecutionRepository(
            dsn,
            schema=schema,
            cipher=ProtectedContextCipher(bytes(range(32))),
            processing_policy=policy,
            finish_policy=policy,
        )
        definition = WorkflowDefinition(
            identifier="empaia-probe",
            inputs={
                "source": WorkflowInputDefinition(
                    kinds=("provider",),
                    media_type="application/octet-stream",
                )
            },
            outputs={
                "result": WorkflowValueDefinition(
                    kind="shared_artifact",
                    media_type="application/octet-stream",
                )
            },
        )
        catalog = WorkflowCatalog(
            (definition,),
            configurations={
                definition.identifier: WorkflowConfiguration(
                    validator=validate_empty_configuration
                )
            },
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"assembled EMPAIA lifecycle")
            service = ControlledEmpaiaService(source)
            provider = EmpaiaProviderAdapter(catalog, service.gateway)
            workflow = WorkflowApplication(
                definition,
                CopyWorkflow(),
                access_adapters={
                    "empaia": EmpaiaInputAccessAdapter(
                        service.gateway,
                        root / "workspace",
                    )
                },
            )
            workflow_app = FastAPI()

            @workflow_app.post("/workflows/empaia-probe/")
            async def run_workflow(invocation: WorkflowRunRequest) -> WorkflowResult:
                """Execute an internal request through the real workflow wrapper, passing private attempt context separately.

                Args:
                    invocation: Named inputs, resolved configuration, public access
                        state, and absolute workflow deadline.

                Returns:
                    WorkflowResult: Named inline outputs and/or references to completed
                        artifact files.
                """
                return await workflow.run(invocation, invocation.execution_context)

            await repository.initialize()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=workflow_app),
                base_url="http://ray-serve",
            ) as workflow_client:
                dispatch = HTTPWorkflowDispatch(
                    workflow_client,
                    origin="http://ray-serve",
                )
                scheduler = ExecutionScheduler(
                    repository=repository,
                    dispatch=InMemoryWorkflowDispatch({}),
                    providers={provider.name: provider},
                    known_workflows=catalog.identifiers,
                    default_timeout=timedelta(minutes=5),
                    maximum_timeout=timedelta(minutes=5),
                )
                worker = ExecutionWorker(
                    repository=repository,
                    active_work=ActiveWorkRegistry(),
                    providers={provider.name: provider},
                    dispatch=dispatch,
                    owner="empaia-integration-worker",
                    lease_duration=timedelta(seconds=30),
                )
                api = create_execution_app(scheduler)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=api),
                    base_url="http://model-service",
                ) as api_client:
                    request = {
                        "installation_id": "installation-1",
                        "api_url": "https://empaia.invalid",
                        "job_id": "job-1",
                        "credential": self._credential(),
                    }
                    scheduled = await api_client.post(
                        "/v1/providers/empaia/workflows/empaia-probe/schedule",
                        json=request,
                    )
                    replay = await api_client.post(
                        "/v1/providers/empaia/workflows/empaia-probe/schedule",
                        json=request,
                    )
                    execution_id = scheduled.json()["execution"]["id"]

                    self.assertTrue(await worker.run_once())
                    self.assertTrue(await worker.run_once())
                    completed = await api_client.get(f"/v1/executions/{execution_id}")

            self.assertEqual(scheduled.status_code, 202)
            self.assertEqual(replay.status_code, 200)
            self.assertEqual(completed.json()["status"], "succeeded")
            self.assertTrue(service.finalized)
            artifact = service.published["result"]
            self.assertIsInstance(artifact, SharedArtifact)
            assert isinstance(artifact, SharedArtifact)
            self.assertEqual(Path(artifact.path).read_bytes(), source.read_bytes())

        async with await AsyncConnection.connect(dsn) as connection:
            await connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )

    async def test_replacement_scheduler_finishes_stored_result_without_redispatch(
        self,
    ) -> None:
        """Verify replacement scheduler finishes stored result without redispatch.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        dsn = os.environ["TEST_POSTGRES_DSN"]
        schema = f"test_empaia_finish_restart_{uuid4().hex}"
        policy = RetryPolicy(3, timedelta(0), timedelta(0))
        repository = PostgresExecutionRepository(
            dsn,
            schema=schema,
            cipher=ProtectedContextCipher(bytes(range(32))),
            processing_policy=policy,
            finish_policy=policy,
        )
        definition = WorkflowDefinition(
            identifier="empaia-probe",
            inputs={
                "source": WorkflowInputDefinition(
                    kinds=("provider",),
                    media_type="application/octet-stream",
                )
            },
            outputs={
                "result": WorkflowValueDefinition(
                    kind="shared_artifact",
                    media_type="application/octet-stream",
                )
            },
        )
        catalog = WorkflowCatalog(
            (definition,),
            configurations={
                definition.identifier: WorkflowConfiguration(
                    validator=validate_empty_configuration
                )
            },
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"finish recovery")
            service = ControlledEmpaiaService(source)
            provider = EmpaiaProviderAdapter(catalog, service.gateway)
            workflow_runs = 0
            workflow = WorkflowApplication(
                definition,
                CopyWorkflow(),
                access_adapters={
                    "empaia": EmpaiaInputAccessAdapter(
                        service.gateway,
                        root / "workspace",
                    )
                },
            )
            workflow_app = FastAPI()

            @workflow_app.post("/workflows/empaia-probe/")
            async def run_workflow(invocation: WorkflowRunRequest) -> WorkflowResult:
                """Execute an internal request through the real workflow wrapper, passing private attempt context separately.

                Args:
                    invocation: Named inputs, resolved configuration, public access
                        state, and absolute workflow deadline.

                Returns:
                    WorkflowResult: Named inline outputs and/or references to completed
                        artifact files.
                """
                nonlocal workflow_runs
                workflow_runs += 1
                return await workflow.run(invocation, invocation.execution_context)

            await repository.initialize()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=workflow_app),
                base_url="http://ray-serve",
            ) as workflow_client:
                dispatch = HTTPWorkflowDispatch(
                    workflow_client,
                    origin="http://ray-serve",
                )
                scheduler = ExecutionScheduler(
                    repository=repository,
                    dispatch=InMemoryWorkflowDispatch({}),
                    providers={provider.name: provider},
                    known_workflows=catalog.identifiers,
                    default_timeout=timedelta(minutes=5),
                    maximum_timeout=timedelta(minutes=5),
                )
                first_worker = ExecutionWorker(
                    repository=repository,
                    active_work=ActiveWorkRegistry(),
                    providers={provider.name: provider},
                    dispatch=dispatch,
                    owner="scheduler-before-restart",
                    lease_duration=timedelta(seconds=30),
                )
                admitted = await scheduler.schedule(
                    ScheduleRequest(
                        workflow_id="empaia-probe",
                        provider="empaia",
                        body={
                            "installation_id": "installation-1",
                            "api_url": "https://empaia.invalid",
                            "job_id": "job-1",
                            "credential": self._credential(),
                        },
                    )
                )

                self.assertTrue(await first_worker.run_once())
                abandoned = await repository.claim_due(
                    "scheduler-before-restart",
                    timedelta(milliseconds=30),
                )
                assert abandoned is not None
                self.assertEqual(abandoned.kind, WorkKind.PROVIDER_FINISH)
                await asyncio.sleep(0.08)

                replacement_repository = PostgresExecutionRepository(
                    dsn,
                    schema=schema,
                    cipher=ProtectedContextCipher(bytes(range(32))),
                    processing_policy=policy,
                    finish_policy=policy,
                )
                replacement_worker = ExecutionWorker(
                    repository=replacement_repository,
                    active_work=ActiveWorkRegistry(),
                    providers={provider.name: provider},
                    dispatch=dispatch,
                    owner="scheduler-after-restart",
                    lease_duration=timedelta(seconds=30),
                )
                self.assertTrue(await replacement_worker.run_once())
                completed = await scheduler.get(admitted.execution.id)

            self.assertEqual(completed.status, "succeeded")
            self.assertEqual(workflow_runs, 1)
            self.assertTrue(service.finalized)
            self.assertEqual(set(service.published), {"result"})

        async with await AsyncConnection.connect(dsn) as connection:
            await connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )

    @staticmethod
    def _credential() -> str:
        """Build an unsigned synthetic JWT-shaped credential with an expiry claim for validation fixtures.

        Returns:
            str: Synthetic unsigned JWT-shaped test credential with an expiration claim.
        """
        header = base64.urlsafe_b64encode(b"{}").rstrip(b"=")
        payload = base64.urlsafe_b64encode(
            json.dumps(
                {"exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp())}
            ).encode()
        ).rstrip(b"=")
        signature = base64.urlsafe_b64encode(b"signature").rstrip(b"=")
        return b".".join((header, payload, signature)).decode()


if __name__ == "__main__":
    unittest.main()
