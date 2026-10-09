"""Demonstrate SDK workflow and raw inference calls against the historical dummy deployment.

The .dat input paths belong to that profile; this demo is not suitable unchanged for the
native full-WSI preset.
"""

from __future__ import annotations

import argparse
import time
from uuid import uuid4

import numpy as np
from rationai import Client
from rationai.types.workflows import (
    ExecutionStatus,
    InputAccess,
    SharedPathInput,
    WorkflowRequest,
)


def workflow_request() -> WorkflowRequest:
    """Create a shared-storage dummy request with a unique TIFF destination and fully resolved configuration.

    Returns:
        WorkflowRequest: Typed SDK request using the selected shared-storage fixture
            inputs.
    """
    return WorkflowRequest(
        inputs={
            "slide": SharedPathInput("input/slide.dat"),
            "region_of_interest": SharedPathInput("input/roi.dat"),
        },
        output_path=f"output/demo-{uuid4().hex}.tiff",
        input_access=InputAccess(
            kind="shared_storage",
            state={"root": "/mnt/validation"},
        ),
    )


def demonstrate_workflows(client: Client) -> None:
    """Schedule and poll a dummy workflow to success, then run an immediate request and print both results.

    Args:
        client: HTTP or SDK client used for requests; resource ownership is described
            above.

    Returns:
        None: No return value.

    Raises:
        RuntimeError: The required lifecycle state or operation result is inconsistent.
        TimeoutError: The described request, cleanup, or polling wait exceeds its time
            budget.
    """
    scheduled = client.workflows.schedule("prostate-heatmap", workflow_request())
    execution_id = scheduled.execution.id
    print(f"Scheduled execution: {execution_id}", flush=True)

    deadline = time.monotonic() + 630
    last_status = None
    while time.monotonic() < deadline:
        execution = client.executions.get(execution_id)
        if execution.status != last_status:
            print(f"  Status: {execution.status}", flush=True)
            last_status = execution.status
        if execution.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            break
        time.sleep(2)
    else:
        raise TimeoutError(
            f"Execution {execution_id} did not finish within 630 seconds; "
            "inspect or cancel it before retrying"
        )

    if execution.status is not ExecutionStatus.SUCCEEDED:
        raise RuntimeError(f"Execution {execution_id} ended with {execution.error}")
    print(f"Scheduled result: {execution.result}", flush=True)

    immediate = client.workflows.run(
        "prostate-heatmap",
        workflow_request(),
        workflow_timeout_seconds=300,
        timeout=330,
    )
    print(f"Immediate result: {immediate}", flush=True)


def demonstrate_inference(client: Client) -> None:
    """Send an all-zero 512-pixel tile through the SDK raw classifier endpoint and print its score.

    Args:
        client: HTTP or SDK client used for requests; resource ownership is described
            above.

    Returns:
        None: No return value.
    """
    tile = np.zeros((512, 512, 3), dtype=np.uint8)
    score = client.models.classify_image("prostate-classifier-1", tile, timeout=900)
    print(f"Classifier score for an all-zero tile: {score}", flush=True)


def main() -> None:
    """Parse the validation origin and selected demo part, then own the SDK client while performing live requests.

    Returns:
        None: No return value.
    """
    parser = argparse.ArgumentParser(
        description="Historical dummy-profile SDK demo; not for the native full-WSI preset."
    )
    parser.add_argument(
        "--origin",
        required=True,
        help="Validation scheduler URL, for example http://127.0.0.1:18080",
    )
    parser.add_argument(
        "--part",
        choices=("all", "workflow", "inference"),
        default="all",
        help="Run both examples or only one (default: all).",
    )
    args = parser.parse_args()

    with Client(base_url=args.origin, timeout=30) as client:
        if args.part in {"all", "workflow"}:
            demonstrate_workflows(client)
        if args.part in {"all", "inference"}:
            demonstrate_inference(client)


if __name__ == "__main__":
    main()
