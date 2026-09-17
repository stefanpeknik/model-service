from __future__ import annotations

import asyncio
import os
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from workflow_execution.contracts import WorkflowInvocation


@unittest.skipUnless(
    os.environ.get("RUN_RAY_SERVE_INTEGRATION") == "1",
    "set RUN_RAY_SERVE_INTEGRATION=1 to run local Ray Serve validation",
)
class RayServeDispatchIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_scale_from_zero_backpressure_and_client_cancellation(self) -> None:
        import httpx
        import ray
        from ray import serve

        from execution_scheduler.dispatch import (
            ConfirmedDispatchRejectionError,
            HTTPWorkflowDispatch,
        )
        from tests.integration.ray_probe_application import ProbeWorkflow
        from workflow_execution.contracts import (
            InlineOutput,
        )

        ray.init(num_cpus=2, include_dashboard=False)
        try:
            with TemporaryDirectory() as directory:
                serve.run(
                    ProbeWorkflow.bind(directory),
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
                        dispatch.run("probe", "scale-from-zero", quick),
                        timeout=120,
                    )
                    value = result.outputs["value"]
                    self.assertIsInstance(value, InlineOutput)
                    assert isinstance(value, InlineOutput)
                    self.assertEqual(value.value, 1)

                    slow = self._invocation(30)
                    active = asyncio.create_task(dispatch.run("probe", "active", slow))
                    await self._wait_for(Path(directory) / "active.started")
                    queued = asyncio.create_task(dispatch.run("probe", "queued", slow))
                    await asyncio.sleep(0.2)
                    with self.assertRaises(ConfirmedDispatchRejectionError):
                        await dispatch.run("probe", "rejected", slow)

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
    async def _wait_for(path: Path) -> None:
        async with asyncio.timeout(30):
            while not path.exists():
                await asyncio.sleep(0.05)


if __name__ == "__main__":
    unittest.main()
