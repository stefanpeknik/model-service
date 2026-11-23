"""Analysis-runner log lines are JSON naming the execution they work for."""

from __future__ import annotations

import json
import logging
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from model_service.analysis_runner.logs import configure_replica_logging
from model_service.analysis_runner.replica import AnalysisReplica
from model_service.analysis_runner.settings import ReplicaSettings
from model_service.catalog import load_catalog
from model_service.catalog.analyses import resolve_run
from model_service.contracts.analysis_http import AnalysisRunRequest
from model_service.contracts.invocation import (
    ExecutionContext,
    InputAccessContext,
    Invocation,
    SharedPathInput,
)
from tests.components.helpers import fake_model, fake_slide
from tests.support.analysis_catalog import REGISTRY, served_model, write
from tests.support.log_capture import capture_logs


if TYPE_CHECKING:
    import numpy as np


class Handle:
    """A served model that logs once it is asked what it runs."""

    def __init__(self, digest: str) -> None:
        self.digest = digest
        self.model = fake_model(tile_size=64, mpp=0.5)

    async def describe(self) -> dict[str, Any]:
        logging.getLogger("model_service.tests.probe").info("halfway")
        return {"id": "m1", "digest": self.digest}

    async def infer(self, tile: np.ndarray) -> np.ndarray:
        return await self.model.infer(tile)


class ReplicaLogTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_runs_log_lines_name_its_execution(self) -> None:
        stream = capture_logs(self, configure_replica_logging)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "catalog", "served-models/m1.yaml", served_model())
            write(
                root / "catalog",
                "analyses/a1.yaml",
                """
                source: {use: slide_tiles, skip_background: false}
                steps: [{model: m1}]
                sinks: [{use: score_grid, output: heatmap}]
                """,
            )
            resolved = resolve_run(
                load_catalog(root / "catalog", registry=REGISTRY), "a1"
            )
            fake_slide(root / "slide.tiff")
            handle = Handle(resolved.model_digests["m1"])
            replica = AnalysisReplica(
                ReplicaSettings(shared_storage_roots=(root,)),
                models=lambda _: handle,
            )
            await replica.run(
                AnalysisRunRequest(
                    Invocation(
                        inputs={"slide": SharedPathInput("slide.tiff")},
                        configuration=None,
                        input_access=InputAccessContext(
                            "shared_storage", {"root": str(root)}
                        ),
                        deadline_at=datetime.now(UTC) + timedelta(minutes=1),
                    ),
                    resolved.document,
                    ExecutionContext(
                        processing_attempt_id="attempt", execution_id="e-1"
                    ),
                )
            )

        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertIn(
            ("halfway", "e-1"),
            [(record["message"], record.get("execution_id")) for record in records],
        )


if __name__ == "__main__":
    unittest.main()
