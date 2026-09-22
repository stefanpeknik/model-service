from __future__ import annotations

import os
import time
import unittest
from uuid import uuid4


class DeployedSDKTests(unittest.TestCase):
    def test_workflow_schedule_poll_and_run_share_one_origin(self) -> None:
        origin = os.environ.get("VALIDATION_ORIGIN")
        if origin is None:
            self.skipTest("VALIDATION_ORIGIN is not configured")

        try:
            from rationai import Client
            from rationai.types.workflows import (
                ExecutionStatus,
                InputAccess,
                SharedPathInput,
                WorkflowRequest,
            )
        except ImportError as error:
            self.skipTest(f"RationAI SDK is unavailable: {error}")

        request = WorkflowRequest(
            inputs={
                "slide": SharedPathInput("input/slide.dat"),
                "region_of_interest": SharedPathInput("input/roi.dat"),
            },
            resolved_configuration={
                "output_path": f"output/{uuid4().hex}.tiff",
                "stride_fraction": 0.5,
                "output_tile_height": 256,
                "output_tile_width": 256,
            },
            input_access=InputAccess(
                kind="shared_storage", state={"root": "/mnt/validation"}
            ),
        )
        with Client(base_url=origin, timeout=30) as client:
            scheduled = client.workflows.schedule("prostate-heatmap", request)
            self.assertTrue(scheduled.created)
            replay = client.workflows.schedule(
                "prostate-heatmap",
                request,
                idempotency_key=scheduled.idempotency_key,
            )
            self.assertFalse(replay.created)
            self.assertEqual(replay.execution.id, scheduled.execution.id)

            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                execution = client.executions.get(scheduled.execution.id)
                if execution.status in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }:
                    break
                time.sleep(1)
            else:
                self.fail("SDK execution polling did not reach a terminal status")

            self.assertEqual(execution.status, ExecutionStatus.SUCCEEDED)
            synchronous = client.workflows.run(
                "prostate-heatmap", request, workflow_timeout_seconds=300
            )
            self.assertEqual(execution.result, synchronous)


if __name__ == "__main__":
    unittest.main()
