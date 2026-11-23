"""EMPAIA inputs reach the analysis runner through the provider's access adapter."""

from __future__ import annotations

import asyncio
import os
import stat
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import FastAPI

from model_service.analysis_runner.http_errors import answer_http
from model_service.analysis_runner.replica import AnalysisReplica
from model_service.analysis_runner.settings import ReplicaSettings
from model_service.api.scheduler.dispatch import (
    HTTPAnalysisDispatch,
    RemoteFailureError,
)
from model_service.catalog import load_catalog
from model_service.catalog.analyses import resolve_run
from model_service.contracts.analysis_http import AnalysisRunRequest
from model_service.contracts.failures import (
    FailureCode,
    InvalidInvocation,
    RunFailure,
)
from model_service.contracts.invocation import (
    INVOCATION,
    ExecutionContext,
    InputAccessContext,
    Invocation,
    ProviderInput,
)
from model_service.providers.empaia.app_service.gateway import EmpaiaGatewayError
from model_service.providers.empaia.app_service.types import DirectAccess
from model_service.providers.empaia.jobs.inputs import EmpaiaInputAccessFactory
from model_service.testing import fake_model, fake_slide
from tests.support.analysis_catalog import REGISTRY, served_model, write
from tests.support.empaia_fixtures import StrictGateway


# A slide on the slide volume, which direct access names.
SLIDE = fake_slide(Path(mkdtemp()) / "slide.tiff")
# The job's public state, with the slide's EAD type admission recorded.
PUBLIC: dict[str, JsonValue] = {
    "api_url": "https://empaia.invalid",
    "job_id": "job",
    "mode": "STANDALONE",
    "inputs": {"slide": {"type": "wsi"}},
}
PROTECTED: dict[str, JsonValue] = {
    "api_url": "https://empaia.invalid",
    "job_id": "job",
    "credential": "secret",
    "publications": {"slide": "slide", "outputs": {}},
}


if TYPE_CHECKING:
    import numpy as np

    from model_service.catalog.analyses import ResolvedAnalysis
    from model_service.contracts.json_values import JsonValue
    from model_service.providers.empaia.app_service.gateway import (
        EmpaiaGateway,
        EmpaiaJobContext,
    )


def invocation(state: dict[str, JsonValue], **inputs: ProviderInput) -> Invocation:
    return Invocation(
        inputs=dict(inputs),
        configuration=None,
        input_access=InputAccessContext("empaia", state),
        deadline_at=datetime.now(UTC) + timedelta(minutes=1),
    )


def slide_job() -> Invocation:
    return invocation(PUBLIC, slide=ProviderInput("slide"))


def attempt(protected: dict[str, JsonValue] | None = None) -> ExecutionContext:
    return ExecutionContext(
        processing_attempt_id="attempt",
        protected_context=PROTECTED if protected is None else protected,
    )


def heatmap(root: Path) -> ResolvedAnalysis:
    """A heatmap analysis over the slide, resolved as admission does."""
    catalog = root / "catalog"
    write(catalog, "served-models/m1.yaml", served_model())
    write(
        catalog,
        "analyses/a1.yaml",
        """
        source: {use: slide_tiles, region_of_interest: ignored}
        steps: [{model: m1}]
        sinks: [{use: score_grid, output: heatmap}]
        """,
    )
    return resolve_run(load_catalog(catalog, registry=REGISTRY), "a1")


class Handle:
    """Served model `m1`, which runs `on_infer` before each tile."""

    def __init__(self, digest: str, on_infer: Any = None) -> None:
        self.digest = digest
        self.model = fake_model(tile_size=64, mpp=0.5)
        self.on_infer = on_infer

    async def describe(self) -> dict[str, Any]:
        return {"id": "m1", "digest": self.digest}

    async def infer(self, tile: np.ndarray) -> np.ndarray:
        if self.on_infer is not None:
            self.on_infer()
        return await self.model.infer(tile)


class EmpaiaInputAccessTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_empaia_acquisition_closes_gateway(self) -> None:
        started = asyncio.Event()
        closed = asyncio.Event()

        class Gateway(StrictGateway):
            async def direct_access(self, name: str) -> DirectAccess | None:
                started.set()
                await asyncio.Event().wait()
                raise AssertionError("the event is never set")

            async def close(self) -> None:
                closed.set()
                raise OSError("cleanup failure")

        with TemporaryDirectory() as directory:
            adapter = EmpaiaInputAccessFactory(lambda _c: Gateway(), Path(directory))
            task = asyncio.create_task(adapter.create(slide_job(), attempt()))
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
            self.assertTrue(closed.is_set())

    async def test_empaia_input_failures_keep_the_gateways_reason(self) -> None:
        cases = [
            EmpaiaGatewayError(
                "EMPAIA AppService is temporarily unavailable", retryable=True
            ),
            EmpaiaGatewayError(
                "EMPAIA slide region is not a readable image", retryable=False
            ),
        ]
        for failure in cases:

            class Gateway(StrictGateway):
                error = failure

                async def direct_access(self, name: str) -> DirectAccess | None:
                    raise self.error

                async def close(self) -> None:
                    return None

            with (
                self.subTest(message=failure.safe_message),
                TemporaryDirectory() as directory,
            ):
                adapter = EmpaiaInputAccessFactory(
                    lambda _c: Gateway(), Path(directory)
                )
                with self.assertRaises(RunFailure) as raised:
                    await adapter.create(slide_job(), attempt())
                self.assertEqual(raised.exception.code, FailureCode.INPUT_UNAVAILABLE)
                self.assertEqual(list(Path(directory).rglob("inputs")), [])
                self.assertEqual(raised.exception.safe_message, failure.safe_message)
                self.assertEqual(raised.exception.retryable, failure.retryable)

    async def test_empaia_access_validates_contexts_before_opening_a_gateway(
        self,
    ) -> None:
        def unexpected_gateway(context: EmpaiaJobContext) -> EmpaiaGateway:
            self.fail("invalid context must not open an AppService gateway")

        public: dict[str, JsonValue] = {
            "api_url": "https://empaia.invalid",
            "job_id": "job",
        }
        protected: dict[str, JsonValue] = {**public, "credential": "secret"}
        runtime = ExecutionContext(
            processing_attempt_id="attempt", protected_context=protected
        )
        cases: list[tuple[dict[str, JsonValue], ExecutionContext]] = [
            ({}, runtime),
            ({**public, "job_id": False}, runtime),
            ({**public, "credential": "secret"}, runtime),
            ({**public, "processing_attempt_id": "attempt"}, runtime),
            ({**public, "job_id": "another-job"}, runtime),
            ({**public, "api_url": "https://another.invalid"}, runtime),
            (public, ExecutionContext(processing_attempt_id="attempt")),
            (public, ExecutionContext(protected_context=protected)),
            (
                public,
                ExecutionContext(
                    processing_attempt_id="attempt",
                    protected_context={**protected, "credential": 1},
                ),
            ),
        ]
        with TemporaryDirectory() as directory:
            adapter = EmpaiaInputAccessFactory(unexpected_gateway, Path(directory))
            for state, context in cases:
                with (
                    self.subTest(state=state),
                    self.assertRaises(InvalidInvocation) as raised,
                ):
                    await adapter.create(invocation(state), context)
                self.assertNotIn("secret", str(raised.exception))

    async def test_dispatch_keeps_runtime_context_out_of_the_durable_invocation(
        self,
    ) -> None:
        seen_credentials: list[str] = []
        closed: list[bool] = []
        seen_requests: list[AnalysisRunRequest] = []

        class Gateway(StrictGateway):
            async def direct_access(self, name: str) -> DirectAccess | None:
                return DirectAccess(SLIDE, supported=True)

            async def close(self) -> None:
                closed.append(True)

        def gateway_factory(context: EmpaiaJobContext) -> Gateway:
            seen_credentials.append(context.credential)
            return Gateway()

        with TemporaryDirectory() as directory:
            resolved = heatmap(Path(directory))
            handle = Handle(resolved.model_digests["m1"])
            core = AnalysisReplica(
                ReplicaSettings(),
                models=lambda _id: handle,
                access={
                    "empaia": EmpaiaInputAccessFactory(
                        gateway_factory, Path(directory) / "workspace"
                    )
                },
            )
            runner = FastAPI()

            @runner.post("/analyses/", response_model=None)
            async def endpoint(body: AnalysisRunRequest):  # type: ignore[no-untyped-def]
                seen_requests.append(body)
                return await answer_http(core.run(body))

            request = slide_job()
            before = INVOCATION.dump_python(request, mode="json")
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=runner)
            ) as client:
                dispatch = HTTPAnalysisDispatch(client, origin="http://ray")
                result = await dispatch.run_analysis(
                    AnalysisRunRequest(request, resolved.document, attempt())
                )
                with self.assertRaises(RemoteFailureError) as raised:
                    await dispatch.run_analysis(
                        AnalysisRunRequest(
                            request,
                            resolved.document,
                            ExecutionContext(processing_attempt_id="attempt"),
                        )
                    )

        self.assertEqual(set(result.outputs), {"heatmap"})
        self.assertEqual(seen_requests[0].execution_context, attempt())
        self.assertEqual(
            INVOCATION.dump_python(seen_requests[0].invocation, mode="json"), before
        )
        self.assertEqual(seen_credentials, ["secret"])
        self.assertEqual(closed, [True])
        self.assertEqual(raised.exception.code, "invalid_run_request")
        self.assertFalse(raised.exception.retryable)
        self.assertNotIn("secret", str(raised.exception))

    async def test_workspace_directories_let_any_user_remove_their_entries(
        self,
    ) -> None:
        # Ray creates the workspace, the API removes it, and they may run as
        # different users: whatever the umask, every directory is 0777.
        modes: dict[str, int] = {}

        class Gateway(StrictGateway):
            async def direct_access(self, name: str) -> DirectAccess | None:
                return DirectAccess(SLIDE, supported=True)

            async def close(self) -> None:
                return None

        with TemporaryDirectory() as directory:
            resolved = heatmap(Path(directory))
            root = Path(directory) / "workspace"

            def record_modes() -> None:
                for entry in root.rglob("*"):
                    if entry.is_dir():
                        modes[str(entry.relative_to(root))] = stat.S_IMODE(
                            entry.stat().st_mode
                        )

            handle = Handle(resolved.model_digests["m1"], record_modes)
            core = AnalysisReplica(
                ReplicaSettings(),
                models=lambda _id: handle,
                access={"empaia": EmpaiaInputAccessFactory(lambda _c: Gateway(), root)},
            )
            previous = os.umask(0o077)
            try:
                await core.run(
                    AnalysisRunRequest(slide_job(), resolved.document, attempt())
                )
            finally:
                os.umask(previous)
            record_modes()

        # The job, its attempt, the inputs and the outputs, at least.
        self.assertGreaterEqual(len(modes), 4)
        self.assertEqual(
            {name: mode for name, mode in modes.items() if mode != 0o777}, {}
        )


if __name__ == "__main__":
    unittest.main()
