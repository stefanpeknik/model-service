"""Builder replica log lines are JSON naming the execution they work for."""

from __future__ import annotations

import json
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from model_service.analysis_runner.logs import configure_replica_logging
from model_service.builders.replica.runner import BuilderApplication
from model_service.contracts.invocation import ExecutionContext
from model_service.contracts.result import BuilderResult, InlineOutput
from model_service.providers.standalone.shared_storage import SharedStorageAccessFactory
from tests.support.builder_fixtures import (
    DECLARED,
    SPATIAL_MODEL,
    FunctionBuilder,
    inline_output,
    invocation,
    spec,
)
from tests.support.log_capture import capture_logs


if TYPE_CHECKING:
    from tests.support.builder_fixtures import AnyContext


class ReplicaLogTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_builders_log_lines_name_its_execution(self) -> None:
        async def build(_context: AnyContext) -> BuilderResult:
            logging.getLogger("model_service.builders.probe").info("halfway")
            return BuilderResult({"score": InlineOutput(1)})

        stream = capture_logs(self, configure_replica_logging)
        with TemporaryDirectory() as directory:
            application = BuilderApplication(
                spec(outputs={"score": inline_output()}),
                FunctionBuilder(build),
                {"shared_storage": SharedStorageAccessFactory([Path(directory)])},
                model_handles=DECLARED,
            )
            await application.run(
                invocation("shared_storage", {"root": directory}),
                SPATIAL_MODEL,
                ExecutionContext(processing_attempt_id="attempt", execution_id="e-1"),
            )

        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertIn(
            ("halfway", "e-1"),
            [(record["message"], record["execution_id"]) for record in records],
        )


if __name__ == "__main__":
    unittest.main()
