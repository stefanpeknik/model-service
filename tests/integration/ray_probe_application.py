from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from ray import serve
from starlette.requests import Request


@serve.deployment(
    num_replicas="auto",
    max_ongoing_requests=1,
    max_queued_requests=1,
    backpressure_config={"status_code": 429, "retry_after_s": 1},
    autoscaling_config={
        "min_replicas": 0,
        "max_replicas": 1,
        "target_ongoing_requests": 1,
    },
)
class ProbeWorkflow:
    def __init__(self, marker_root: str) -> None:
        self._marker_root = Path(marker_root)

    async def __call__(self, request: Request) -> dict[str, object]:
        attempt = request.headers.get("x-processing-attempt-id", f"synchronous-{uuid4()}")
        body = await request.json()
        delay = body["resolved_configuration"]["delay_seconds"]
        (self._marker_root / f"{attempt}.started").touch()
        try:
            await asyncio.sleep(delay)
        finally:
            (self._marker_root / f"{attempt}.stopped").touch()
        return {"outputs": {"value": {"kind": "inline", "value": 1}}}
