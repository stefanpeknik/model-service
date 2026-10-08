"""Check assembled PostgreSQL/API dispatch and backpressure/cancellation using a local Ray Serve probe.

Opt-in integration checks start a local Ray cluster and can also require PostgreSQL.
Filesystem markers distinguish acceptance from cancellation.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from base64 import urlsafe_b64encode
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from uuid import uuid4


if TYPE_CHECKING:
    from workflow_execution.contracts import WorkflowInvocation


# Ray otherwise reproduces the parent `uv run` environment for every local worker.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


@unittest.skipUnless(
    os.environ.get("RUN_RAY_SERVE_INTEGRATION") == "1",
    "set RUN_RAY_SERVE_INTEGRATION=1 to run local Ray Serve validation",
)
class RayServeDispatchIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Exercise assembled PostgreSQL/API dispatch and backpressure/cancellation using a local Ray Serve probe."""

    async def test_public_api_postgres_worker_and_ray_serve_are_assembled(
        self,
    ) -> None:
        """Verify public API PostgreSQL worker and Ray serve are assembled.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        dsn = os.environ.get("TEST_POSTGRES_DSN")
        if dsn is None:
            self.skipTest("TEST_POSTGRES_DSN is not configured")

        import httpx
        import ray
        from psycopg import AsyncConnection, sql
        from ray import serve

        from execution_scheduler.contracts import ExecutionStatus
        from execution_scheduler.domain import RetryPolicy
        from execution_scheduler.persistence.postgres import PostgresExecutionRepository
        from execution_scheduler.persistence.protected_context import (
            ProtectedContextCipher,
        )
        from execution_scheduler.scheduler import ExecutionScheduler
        from integrations.standalone import StandaloneProviderAdapter
        from model_service.api import create_execution_app
        from model_service.synchronous import SynchronousWorkflowService
        from tests.integration.ray_probe_application import ProbeWorkflow
        from workflow_execution.catalog import (
            WorkflowCatalog,
            WorkflowConfiguration,
            WorkflowDefinition,
            WorkflowValueDefinition,
            validate_empty_configuration,
        )
        from workflow_execution.dispatch import HTTPWorkflowDispatch

        schema = f"test_assembled_{uuid4().hex}"
        policy = RetryPolicy(3, timedelta(0), timedelta(0))
        repository = PostgresExecutionRepository(
            dsn,
            schema=schema,
            cipher=ProtectedContextCipher.from_base64(
                urlsafe_b64encode(bytes(range(32))).decode()
            ),
            processing_policy=policy,
            finish_policy=policy,
        )
        definition = WorkflowDefinition(
            identifier="probe",
            inputs={},
            outputs={"value": WorkflowValueDefinition(kind="inline")},
        )
        catalog = WorkflowCatalog(
            (definition,),
            configurations={
                definition.identifier: WorkflowConfiguration(
                    validator=validate_empty_configuration
                )
            },
        )
        provider = StandaloneProviderAdapter(catalog)
        ray.init(num_cpus=2, include_dashboard=False)
        try:
            with TemporaryDirectory() as directory:
                await repository.initialize()
                serve.run(
                    ProbeWorkflow.bind(directory),  # type: ignore[attr-defined]
                    name="probe",
                    route_prefix="/workflows/probe",
                )
                async with httpx.AsyncClient() as ray_client:
                    dispatch = HTTPWorkflowDispatch(
                        ray_client,
                        origin="http://127.0.0.1:8000",
                    )
                    scheduler = ExecutionScheduler(
                        repository=repository,
                        dispatch=dispatch,
                        providers={provider.name: provider},
                        known_workflows=catalog.identifiers,
                        default_timeout=timedelta(minutes=3),
                        maximum_timeout=timedelta(minutes=3),
                        lease_duration=timedelta(seconds=30),
                        poll_interval=0.01,
                    )
                    app = create_execution_app(
                        scheduler,
                        workflow_runner=SynchronousWorkflowService(
                            catalog,
                            dispatch,
                        ),
                    )
                    async with (
                        scheduler.running(),
                        httpx.AsyncClient(
                            transport=httpx.ASGITransport(app=app),
                            base_url="http://model-service",
                        ) as api_client,
                    ):
                        invocation = self._standalone_invocation(directory)
                        scheduled = await api_client.post(
                            "/v1/workflows/probe/schedule",
                            json={
                                "provider": "standalone",
                                "body": invocation,
                                "idempotency_key": "assembled-probe",
                            },
                        )
                        replay = await api_client.post(
                            "/v1/workflows/probe/schedule",
                            json={
                                "provider": "standalone",
                                "body": invocation,
                                "idempotency_key": "assembled-probe",
                            },
                        )
                        execution_id = scheduled.json()["execution"]["id"]

                        async with asyncio.timeout(30):
                            while True:
                                completed = await api_client.get(
                                    f"/v1/executions/{execution_id}"
                                )
                                if completed.json()["status"] in {
                                    ExecutionStatus.SUCCEEDED,
                                    ExecutionStatus.FAILED,
                                    ExecutionStatus.CANCELLED,
                                }:
                                    break
                                await asyncio.sleep(0.01)
                        synchronous = await api_client.post(
                            "/v1/workflows/probe/run",
                            json={
                                **invocation,
                                "deadline_at": (
                                    datetime.now(UTC) + timedelta(minutes=3)
                                ).isoformat(),
                            },
                        )

                self.assertEqual(scheduled.status_code, 202)
                self.assertEqual(replay.status_code, 200)
                self.assertEqual(completed.json()["status"], "succeeded")
                self.assertEqual(
                    completed.json()["result"],
                    {"outputs": {"value": {"kind": "inline", "value": 1}}},
                )
                self.assertEqual(synchronous.status_code, 200)
                self.assertEqual(synchronous.json(), completed.json()["result"])
        finally:
            serve.shutdown()
            ray.shutdown()
            async with await AsyncConnection.connect(dsn) as connection:
                await connection.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                        sql.Identifier(schema)
                    )
                )

    async def test_scale_from_zero_backpressure_and_client_cancellation(self) -> None:
        """Verify scale from zero backpressure and client cancellation.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        import httpx
        import ray
        from ray import serve

        from tests.integration.ray_probe_application import ProbeWorkflow
        from workflow_execution.contracts import (
            InlineOutput,
            WorkflowExecutionContext,
        )
        from workflow_execution.dispatch import (
            ConfirmedDispatchRejectionError,
            HTTPWorkflowDispatch,
        )

        ray.init(num_cpus=2, include_dashboard=False)
        try:
            with TemporaryDirectory() as directory:
                serve.run(
                    ProbeWorkflow.bind(directory),  # type: ignore[attr-defined]
                    name="probe",
                    route_prefix="/workflows/probe",
                )
                async with httpx.AsyncClient() as client:
                    dispatch = HTTPWorkflowDispatch(
                        client,
                        origin="http://127.0.0.1:8000",
                    )
                    quick = self._invocation(0)
                    result = await asyncio.wait_for(
                        dispatch.run(
                            "probe",
                            WorkflowExecutionContext(
                                processing_attempt_id="scale-from-zero"
                            ),
                            quick,
                        ),
                        timeout=120,
                    )
                    value = result.outputs["value"]
                    self.assertIsInstance(value, InlineOutput)
                    assert isinstance(value, InlineOutput)
                    self.assertEqual(value.value, 1)

                    slow = self._invocation(30)
                    active = asyncio.create_task(
                        dispatch.run(
                            "probe",
                            WorkflowExecutionContext(processing_attempt_id="active"),
                            slow,
                        )
                    )
                    await self._wait_for(Path(directory) / "active.started")
                    queued = asyncio.create_task(
                        dispatch.run(
                            "probe",
                            WorkflowExecutionContext(processing_attempt_id="queued"),
                            slow,
                        )
                    )
                    await asyncio.sleep(0.2)
                    with self.assertRaises(ConfirmedDispatchRejectionError):
                        await dispatch.run(
                            "probe",
                            WorkflowExecutionContext(processing_attempt_id="rejected"),
                            slow,
                        )

                    active.cancel()
                    await asyncio.gather(active, return_exceptions=True)
                    await self._wait_for(Path(directory) / "active.stopped")
                    queued.cancel()
                    await asyncio.gather(queued, return_exceptions=True)
        finally:
            serve.shutdown()
            ray.shutdown()

    @staticmethod
    def _invocation(delay_seconds: int) -> WorkflowInvocation:
        """Build a typed workflow invocation with the fixture access context and a bounded future deadline.

        Args:
            delay_seconds: Artificial computation delay used to observe acceptance and
                cancellation.

        Returns:
            WorkflowInvocation: Typed inputs, resolved configuration, access settings,
                and absolute deadline.
        """
        from workflow_execution.contracts import (
            InputAccessContext,
            WorkflowInvocation,
        )

        return WorkflowInvocation(
            inputs={},
            resolved_configuration={"delay_seconds": delay_seconds},
            input_access=InputAccessContext(kind="integration", state={}),
            deadline_at=datetime.now(UTC) + timedelta(minutes=3),
        )

    @staticmethod
    def _standalone_invocation(directory: str) -> dict[str, object]:
        """Build a minimal shared-storage JSON body for scheduling the delayed Ray probe.

        Args:
            directory: Directory containing definitions or generated validation
                fixtures.

        Returns:
            dict[str, object]: Decoded, serialized, or recorded fields described above.
        """
        return {
            "inputs": {},
            "resolved_configuration": {"delay_seconds": 0},
            "input_access": {
                "kind": "shared_storage",
                "state": {"root": directory},
            },
        }

    @staticmethod
    async def _wait_for(path: Path) -> None:
        """Wait at most 30 seconds for a Ray probe marker to appear.

        Args:
            path: File, storage locator, or HTTP endpoint selected by this operation.

        Returns:
            None: No return value.
        """
        async with asyncio.timeout(30):
            while not path.exists():
                await asyncio.sleep(0.05)


if __name__ == "__main__":
    unittest.main()
