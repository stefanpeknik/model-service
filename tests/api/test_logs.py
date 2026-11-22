"""API log lines are JSON naming their HTTP request and, where there is one, execution."""

from __future__ import annotations

import io
import json
import logging
import logging.config
import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from httpx import ASGITransport, AsyncClient

import model_service.api
from model_service.api.app_factory import create_app
from model_service.api.scheduler.execution_scheduler import ExecutionScheduler
from model_service.api.scheduler.running_work import ActiveWorkRegistry
from model_service.api.scheduler.worker import ExecutionWorker
from model_service.catalog import Catalog
from model_service.contracts.result import BuilderResult, InlineOutput
from model_service.providers.standalone.adapter import StandaloneProviderAdapter
from model_service.providers.standalone.shared_storage import SharedStorageAccessFactory
from tests.support.log_capture import capture_logs
from tests.support.memory_repository import (
    FakeDispatch,
    InMemoryExecutionRepository,
)
from tests.support.scheduling import resolve_test_analysis, scheduling_catalog


if TYPE_CHECKING:
    from model_service.contracts.analysis_http import AnalysisRunRequest


# What Uvicorn loads with `--log-config` in the API container.
LOG_CONFIG = Path(model_service.api.__file__).with_name("log-config.json")
# Scheduling an analysis the catalog lacks is refused, and the refusal is logged.
UNKNOWN_ANALYSIS = "/v1/analyses/missing/schedule"
BODY = {
    "inputs": {"slide": {"kind": "shared_path", "path": "slide.tiff"}},
    "input_access": {"kind": "shared_storage", "state": {"root": "/data"}},
}


class RequestIdTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        catalog = Catalog(None)
        scheduler = ExecutionScheduler(
            repository=InMemoryExecutionRepository(),
            dispatch=FakeDispatch({}),
            providers={
                "standalone": StandaloneProviderAdapter(
                    SharedStorageAccessFactory([Path("/data")])
                )
            },
            resolve_analysis=resolve_test_analysis,
            default_timeout=timedelta(minutes=1),
            maximum_timeout=timedelta(hours=1),
        )
        self.client = AsyncClient(
            transport=ASGITransport(app=create_app(scheduler, catalog=lambda: catalog)),
            base_url="http://api",
        )
        self.addAsyncCleanup(self.client.aclose)

    def capture_logs(self) -> io.StringIO:
        """Configure logging as the API container does, writing to a buffer."""
        return capture_logs(
            self,
            lambda: logging.config.dictConfig(json.loads(LOG_CONFIG.read_text())),
        )

    async def test_a_callers_request_id_is_returned(self) -> None:
        response = await self.client.get(
            "/v1/models", headers={"X-Request-ID": "jes-1234"}
        )

        self.assertEqual(response.headers["X-Request-ID"], "jes-1234")

    async def test_a_request_without_an_id_gets_one(self) -> None:
        first = await self.client.get("/v1/models")
        second = await self.client.get("/v1/models")

        self.assertTrue(first.headers["X-Request-ID"])
        self.assertNotEqual(
            first.headers["X-Request-ID"], second.headers["X-Request-ID"]
        )

    async def test_log_lines_are_json_naming_their_request(self) -> None:
        stream = self.capture_logs()

        response = await self.client.post(
            UNKNOWN_ANALYSIS,
            json=BODY,
            headers={"X-Request-ID": "jes-1234", "Idempotency-Key": "k"},
        )

        self.assertEqual(response.status_code, 404, response.text)
        (line,) = stream.getvalue().splitlines()
        record = json.loads(line)
        self.assertEqual(record["request_id"], "jes-1234")
        self.assertEqual(record["level"], "WARNING")
        self.assertEqual(record["logger"], "model_service.api.http.scheduling")


class ExecutionIdTests(unittest.IsolatedAsyncioTestCase):
    async def test_log_lines_while_an_execution_runs_name_it(self) -> None:
        class LoggingDispatch:
            async def run_analysis(self, request: AnalysisRunRequest) -> BuilderResult:
                logging.getLogger("model_service.tests").warning("dispatching")
                return BuilderResult({"heatmap": InlineOutput(1)})

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "slide.tiff").touch()
            repository = InMemoryExecutionRepository()
            providers = {
                "standalone": StandaloneProviderAdapter(
                    SharedStorageAccessFactory([root])
                )
            }
            scheduler = ExecutionScheduler(
                repository=repository,
                dispatch=LoggingDispatch(),
                providers=providers,
                resolve_analysis=resolve_test_analysis,
                default_timeout=timedelta(minutes=1),
                maximum_timeout=timedelta(hours=1),
            )
            worker = ExecutionWorker(
                repository=repository,
                active_work=ActiveWorkRegistry(),
                providers=providers,
                dispatch=LoggingDispatch(),
                owner="worker-1",
                lease_duration=timedelta(minutes=1),
            )
            async with AsyncClient(
                transport=ASGITransport(
                    app=create_app(scheduler, catalog=scheduling_catalog)
                ),
                base_url="http://api",
            ) as client:
                response = await client.post(
                    "/v1/analyses/a1/schedule",
                    json={
                        **BODY,
                        "input_access": {
                            "kind": "shared_storage",
                            "state": {"root": str(root)},
                        },
                    },
                    headers={"Idempotency-Key": "k"},
                )
            self.assertEqual(response.status_code, 202, response.text)
            stream = capture_logs(
                self,
                lambda: logging.config.dictConfig(json.loads(LOG_CONFIG.read_text())),
            )

            await worker.run_once()

        (line,) = stream.getvalue().splitlines()
        self.assertEqual(
            json.loads(line)["execution_id"], response.json()["execution"]["id"]
        )


if __name__ == "__main__":
    unittest.main()
