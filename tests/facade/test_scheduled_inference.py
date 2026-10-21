"""Scheduled raw inference through the public API, scheduler, worker and a fake Ray model."""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import httpx
import lz4.frame
from httpx import ASGITransport, AsyncClient

from model_service.builders.contracts import (
    BuilderResult,
    InputAccessContext,
    Invocation,
    SpatialScoresModel,
    TileEmbeddingModel,
)
from model_service.facade.api import create_app
from model_service.facade.app import model_resolver
from model_service.facade.catalog import ModelCatalog
from model_service.facade.inference import (
    HttpInferenceProxy,
    ModelInference,
    check_tile,
)
from model_service.providers.shared_storage import SharedStorageAccessAdapter
from model_service.providers.standalone import StandaloneProviderAdapter
from model_service.scheduler.domain import Admission, RetryPolicy, SubmissionIdentity
from model_service.scheduler.inference import IncompatibleInputError
from model_service.scheduler.scheduler import ExecutionScheduler
from tests.support.memory_repository import FakeDispatch, InMemoryExecutionRepository


if TYPE_CHECKING:
    from collections.abc import Callable

    from model_service.builders.contracts import ExecutionContext


CLASSIFIER = SpatialScoresModel(
    id="classifier", tile_size=4, output_tile_size=1, n_channels=1, mpp=0.5
)
EMBEDDER = TileEmbeddingModel(id="embedder", tile_size=2, mpp=0.5, summaries=("g",))
CATALOG = ModelCatalog([CLASSIFIER, EMBEDDER])


def tile(size: int = 4) -> bytes:
    return lz4.frame.compress(bytes(range(size * size * 3)))


type ModelHandler = Callable[[httpx.Request], httpx.Response]


class ScheduledInferenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.model_requests: list[httpx.Request] = []
        self.model_handler: ModelHandler = lambda _request: httpx.Response(
            200,
            content=b"scores",
            headers={
                "content-type": "application/octet-stream",
                "x-output-shape": "(1,)",
            },
        )

        def ray(request: httpx.Request) -> httpx.Response:
            self.model_requests.append(request)
            return self.model_handler(request)

        self.http = httpx.AsyncClient(transport=httpx.MockTransport(ray))
        self.addAsyncCleanup(self.http.aclose)
        policy = RetryPolicy(3, timedelta(0), timedelta(0))
        self.repository = InMemoryExecutionRepository(
            processing_policy=policy, finish_policy=policy
        )
        self.dispatch = FakeDispatch({"heatmap": self._builder})
        self.builder_release = asyncio.Event()
        self.builder_release.set()
        self.target = ModelInference(
            CATALOG,
            HttpInferenceProxy(self.http, origin="http://ray"),
            max_result_bytes=64,
        )
        self.scheduler = ExecutionScheduler(
            repository=self.repository,
            dispatch=self.dispatch,
            providers={
                "standalone": StandaloneProviderAdapter(SharedStorageAccessAdapter([]))
            },
            resolve_model=model_resolver(CATALOG),
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            poll_interval=0.01,
            inference_target=self.target,
            max_concurrent_inference=4,
        )
        self.api = AsyncClient(
            transport=ASGITransport(
                app=create_app(
                    self.scheduler, catalog=CATALOG, inference_max_request_bytes=1024
                )
            ),
            base_url="http://model-service",
        )
        self.addAsyncCleanup(self.api.aclose)

    async def _builder(
        self, _invocation: Invocation, _context: ExecutionContext
    ) -> BuilderResult:
        await self.builder_release.wait()
        return BuilderResult({})

    async def schedule(
        self,
        key: str = "tile-1",
        payload: bytes | None = None,
        model_id: str = CLASSIFIER.id,
        **headers: str,
    ) -> httpx.Response:
        return await self.api.post(
            f"/v1/models/{model_id}/schedule",
            content=tile() if payload is None else payload,
            headers={"idempotency-key": key, **headers},
        )

    async def run_attempt(self) -> None:
        self.assertTrue(await self.scheduler._worker.run_once())

    async def execution(self, execution_id: str) -> dict[str, object]:
        response = await self.api.get(f"/v1/executions/{execution_id}")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def assert_error(self, response: httpx.Response, status: int, code: str) -> None:
        self.assertEqual(response.status_code, status, response.text)
        self.assertEqual(response.json()["detail"]["code"], code)

    async def test_scheduled_tile_runs_once_and_its_response_is_downloadable(
        self,
    ) -> None:
        created = await self.schedule(**{"x-output-dtype": "float16"})
        replay = await self.schedule(**{"x-output-dtype": "float16"})
        execution_id = created.json()["execution"]["id"]

        self.assertEqual(created.status_code, 202, created.text)
        self.assertEqual(replay.status_code, 200)
        self.assertEqual(replay.json()["execution"]["id"], execution_id)
        self.assertEqual(
            created.json()["execution"],
            {
                "id": execution_id,
                "operation": "inference",
                "builder_id": None,
                "model_id": "classifier",
                "status": "queued",
                "result": None,
                "error": None,
                "provider_pending": False,
                "message": None,
            },
        )

        await self.run_attempt()

        [request] = self.model_requests
        self.assertEqual(str(request.url), "http://ray/classifier/")
        self.assertEqual(request.content, tile())
        self.assertEqual(request.headers["x-output-dtype"], "float16")
        view = await self.execution(execution_id)
        self.assertEqual(view["status"], "succeeded")
        self.assertEqual(
            view["result"],
            {
                "media_type": "application/octet-stream",
                "headers": {"x-output-shape": "(1,)"},
                "size": 6,
            },
        )
        result = await self.api.get(f"/v1/executions/{execution_id}/result")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.content, b"scores")
        self.assertEqual(result.headers["x-output-shape"], "(1,)")
        self.assertEqual(result.headers["content-type"], "application/octet-stream")

    async def test_admission_checks_key_model_tile_size_and_replay_identity(
        self,
    ) -> None:
        missing_key = await self.api.post(
            "/v1/models/classifier/schedule", content=tile()
        )
        self.assert_error(missing_key, 422, "invalid_request")
        self.assert_error(await self.schedule(model_id="missing"), 404, "unknown_model")
        self.assert_error(
            await self.schedule(payload=tile(size=2)), 422, "invalid_request"
        )
        self.assert_error(
            await self.schedule(payload=b"not lz4"), 422, "invalid_request"
        )
        self.assert_error(
            await self.schedule(payload=bytes(2048)), 413, "payload_too_large"
        )
        self.assertEqual((await self.schedule("reused")).status_code, 202)
        self.assert_error(
            await self.schedule("reused", **{"x-output-dtype": "float16"}),
            409,
            "schedule_conflict",
        )
        self.assertEqual(self.model_requests, [])

    async def test_backpressure_requeues_without_spending_an_attempt(self) -> None:
        responses = iter(
            [httpx.Response(503, headers={"retry-after": "0"}), httpx.Response(200)]
        )
        self.model_handler = lambda _request: next(responses)
        execution_id = (await self.schedule()).json()["execution"]["id"]

        await self.run_attempt()
        queued = await self.repository.get(execution_id)
        await self.run_attempt()

        self.assertEqual(
            (queued.status.value, queued.processing_attempts), ("queued", 0)
        )
        self.assertEqual((await self.execution(execution_id))["status"], "succeeded")

    async def test_model_failures_are_classified(self) -> None:
        def unreachable(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        cases: list[tuple[ModelHandler, int, str]] = [
            (lambda _request: httpx.Response(400), 1, "model_rejected_input"),
            (lambda _request: httpx.Response(500), 3, "model_failed"),
            (unreachable, 3, "model_unavailable"),
            (
                lambda _request: httpx.Response(200, content=bytes(65)),
                1,
                "inference_result_too_large",
            ),
        ]
        for index, (handler, attempts, code) in enumerate(cases):
            with self.subTest(code=code):
                self.model_handler = handler
                execution_id = (await self.schedule(f"tile-{index}")).json()[
                    "execution"
                ]["id"]
                for _ in range(attempts):
                    await self.run_attempt()
                self.assertFalse(await self.scheduler._worker.run_once())
                view = await self.execution(execution_id)
                self.assertEqual(view["status"], "failed")
                self.assertEqual(view["error"]["code"], code)
                self.assert_error(
                    await self.api.get(f"/v1/executions/{execution_id}/result"),
                    409,
                    "result_unavailable",
                )

    async def test_a_model_change_fails_a_queued_tile_instead_of_reinterpreting_it(
        self,
    ) -> None:
        execution_id = (await self.schedule()).json()["execution"]["id"]
        resized = SpatialScoresModel(
            id="classifier", tile_size=8, output_tile_size=1, n_channels=1, mpp=0.5
        )
        self.target._catalog = ModelCatalog([resized, EMBEDDER])

        await self.run_attempt()

        view = await self.execution(execution_id)
        self.assertEqual(view["error"]["code"], "incompatible_input")
        self.assertEqual(self.model_requests, [])

    async def test_cancelled_tile_is_never_sent(self) -> None:
        execution_id = (await self.schedule()).json()["execution"]["id"]

        cancelled = await self.api.post(f"/v1/executions/{execution_id}/cancel")

        self.assertEqual(cancelled.json()["status"], "cancelled")
        self.assertFalse(await self.scheduler._worker.run_once())
        self.assertEqual(self.model_requests, [])
        self.assertNotIn(execution_id, self.repository.requests)

    async def test_lookup_and_result_errors(self) -> None:
        first = (await self.schedule("first")).json()["execution"]["id"]
        second = (await self.schedule("second")).json()["execution"]["id"]

        found = await self.api.post(
            "/v1/executions/lookup", json={"ids": [second, "unknown", first]}
        )
        too_many = await self.api.post(
            "/v1/executions/lookup", json={"ids": ["x"] * 1001}
        )

        self.assertEqual([view["id"] for view in found.json()], [second, first])
        self.assert_error(too_many, 422, "invalid_request")
        self.assert_error(
            await self.api.get(f"/v1/executions/{first}/result"),
            409,
            "result_unavailable",
        )
        self.assert_error(
            await self.api.get("/v1/executions/unknown/result"),
            404,
            "unknown_execution",
        )

    async def test_tiles_and_builders_use_separate_worker_slots(self) -> None:
        self.builder_release.clear()
        async with self.scheduler.running():
            builder_id = await self._admit_builder()
            await _wait_for(lambda: self.dispatch.models != [])
            execution_id = (await self.schedule()).json()["execution"]["id"]
            await _wait_for_status(self, execution_id, "succeeded")
            self.assertEqual((await self.execution(builder_id))["status"], "running")
            self.builder_release.set()
            await _wait_for_status(self, builder_id, "succeeded")

    async def _admit_builder(self) -> str:
        deadline = datetime.now(UTC) + timedelta(minutes=5)
        admitted = await self.repository.admit(
            Admission(
                identity=SubmissionIdentity("standalone", "builder", "builder"),
                builder_id="heatmap",
                model_id=CLASSIFIER.id,
                invocation=Invocation(
                    inputs={},
                    resolved_configuration=None,
                    input_access=InputAccessContext(
                        kind="shared_storage", state={"root": "/"}
                    ),
                    deadline_at=deadline,
                ),
                protected_context={},
                deadline_at=deadline,
            )
        )
        return admitted.execution.id


class TileCheckTests(unittest.TestCase):
    def test_exact_lz4_tile_of_the_model_size_is_accepted(self) -> None:
        check_tile(CLASSIFIER, tile())
        check_tile(EMBEDDER, tile(size=2))

    def test_other_payloads_are_incompatible(self) -> None:
        cases = {
            "smaller": tile(size=2),
            "larger": tile(size=8),
            "trailing data": tile() + b"x",
            "truncated": tile()[:-4],
            "not lz4": b"raw",
        }
        for name, payload in cases.items():
            with self.subTest(name), self.assertRaises(IncompatibleInputError):
                check_tile(CLASSIFIER, payload)


async def _wait_for(condition: Callable[[], bool], timeout: float = 2) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


async def _wait_for_status(
    test: ScheduledInferenceTests, execution_id: str, status: str
) -> None:
    async with asyncio.timeout(2):
        while (await test.execution(execution_id))["status"] != status:
            await asyncio.sleep(0.01)


if __name__ == "__main__":
    unittest.main()
