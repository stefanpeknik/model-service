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

from execution_scheduler.api import create_execution_app
from execution_scheduler.contracts import ScheduleRequest
from execution_scheduler.dispatch import HTTPWorkflowDispatch
from execution_scheduler.domain import RetryPolicy, WorkKind
from execution_scheduler.postgres import PostgresExecutionRepository
from execution_scheduler.scheduler import ExecutionScheduler
from execution_scheduler.security import ProtectedContextCipher
from execution_scheduler.worker import ExecutionWorker
from integrations.empaia.input_access import EmpaiaInputAccessAdapter
from integrations.empaia.provider import EmpaiaProviderAdapter
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
from workflow_execution.http import (
    WorkflowRunRequest,  # noqa: TC001 - FastAPI resolves endpoint annotations at runtime.
)


if TYPE_CHECKING:
    from integrations.empaia.gateway import EmpaiaJobContext


@dataclass(slots=True)
class ControlledEmpaiaService:
    source: Path
    published: dict[str, WorkflowOutput] = field(default_factory=dict)
    finalized: bool = False

    def gateway(self, context: EmpaiaJobContext) -> ControlledEmpaiaGateway:
        del context
        return ControlledEmpaiaGateway(self)


class ControlledEmpaiaGateway:
    def __init__(self, service: ControlledEmpaiaService) -> None:
        self._service = service

    async def get_configuration(self) -> dict[str, JsonValue]:
        return {}

    async def materialize_input(self, _name: str, destination: str) -> None:
        shutil.copyfile(self._service.source, destination)

    async def publish_output(self, name: str, output: WorkflowOutput) -> None:
        self._service.published[name] = output

    async def finalize(self) -> None:
        self._service.finalized = True

    async def report_failure(self, message: str) -> None:
        raise AssertionError(message)

    async def close(self) -> None:
        return None


class CopyWorkflow:
    async def run(self, context: WorkflowContext) -> WorkflowResult:
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
    async def test_public_empaia_lifecycle_crosses_real_persistence_and_http(
        self,
    ) -> None:
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
                    providers={provider.name: provider},
                    known_workflows=catalog.identifiers,
                    default_timeout=timedelta(minutes=5),
                    maximum_timeout=timedelta(minutes=5),
                )
                worker = ExecutionWorker(
                    repository=repository,
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
                    providers={provider.name: provider},
                    known_workflows=catalog.identifiers,
                    default_timeout=timedelta(minutes=5),
                    maximum_timeout=timedelta(minutes=5),
                )
                first_worker = ExecutionWorker(
                    repository=repository,
                    providers={provider.name: provider},
                    dispatch=dispatch,
                    owner="scheduler-before-restart",
                    lease_duration=timedelta(milliseconds=30),
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
        header = base64.urlsafe_b64encode(b"{}").rstrip(b"=")
        payload = base64.urlsafe_b64encode(
            json.dumps(
                {"exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp())}
            ).encode()
        ).rstrip(b"=")
        return b".".join((header, payload, b"signature")).decode()


if __name__ == "__main__":
    unittest.main()
