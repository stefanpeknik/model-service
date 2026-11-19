"""Every API response and log line names the HTTP request it belongs to."""

from __future__ import annotations

import io
import json
import logging
import logging.config
import unittest
from contextlib import redirect_stdout
from datetime import timedelta
from pathlib import Path

from httpx import ASGITransport, AsyncClient

import model_service.api
from model_service.api.app_factory import create_app, model_resolver
from model_service.api.model_catalog import ModelCatalog
from model_service.api.scheduler.execution_scheduler import ExecutionScheduler
from model_service.providers.standalone.adapter import StandaloneProviderAdapter
from model_service.providers.standalone.shared_storage import SharedStorageAccessFactory
from tests.support.memory_repository import InMemoryExecutionRepository


# What Uvicorn loads with `--log-config` in the API container.
LOG_CONFIG = Path(model_service.api.__file__).with_name("log-config.json")
# Scheduling a model the catalog lacks is refused, and the refusal is logged.
UNKNOWN_MODEL = "/v1/builders/heatmap/models/missing/schedule"
BODY = {
    "inputs": {"slide": {"kind": "shared_path", "path": "slide.tiff"}},
    "input_access": {"kind": "shared_storage", "state": {"root": "/data"}},
}


class RequestIdTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        catalog = ModelCatalog([])
        scheduler = ExecutionScheduler(
            repository=InMemoryExecutionRepository(),
            dispatch=None,  # type: ignore[arg-type]
            providers={
                "standalone": StandaloneProviderAdapter(
                    SharedStorageAccessFactory([Path("/data")])
                )
            },
            resolve_model=model_resolver(catalog),
            default_timeout=timedelta(minutes=1),
            maximum_timeout=timedelta(hours=1),
        )
        self.client = AsyncClient(
            transport=ASGITransport(app=create_app(scheduler, catalog=catalog)),
            base_url="http://api",
        )
        self.addAsyncCleanup(self.client.aclose)

    def capture_logs(self) -> io.StringIO:
        """Configure logging as the API container does, writing to a buffer."""
        configured = ["", "model_service", "uvicorn", "uvicorn.access"]
        saved = [
            (logger, logger.handlers[:], logger.level, logger.propagate)
            for logger in map(logging.getLogger, configured)
        ]

        def restore() -> None:
            for logger, handlers, level, propagate in saved:
                logger.handlers, logger.level, logger.propagate = (
                    handlers,
                    level,
                    propagate,
                )

        self.addCleanup(restore)
        stream = io.StringIO()
        with redirect_stdout(stream):
            logging.config.dictConfig(json.loads(LOG_CONFIG.read_text()))
        return stream

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
            UNKNOWN_MODEL,
            json=BODY,
            headers={"X-Request-ID": "jes-1234", "Idempotency-Key": "k"},
        )

        self.assertEqual(response.status_code, 404, response.text)
        (line,) = stream.getvalue().splitlines()
        record = json.loads(line)
        self.assertEqual(record["request_id"], "jes-1234")
        self.assertEqual(record["level"], "WARNING")
        self.assertEqual(record["logger"], "model_service.api.http.scheduling")


if __name__ == "__main__":
    unittest.main()
