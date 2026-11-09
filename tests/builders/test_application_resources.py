"""Check deadline-aware acquisition, bounded cleanup and cancellation."""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from model_service.builders.application import BuilderApplication
from model_service.builders.contracts import (
    BuilderFailure,
    BuilderResult,
    InvalidInvocation,
)
from tests.support.builder_fixtures import (
    SPATIAL_MODEL,
    FunctionBuilder,
    invocation,
    spec,
)


if TYPE_CHECKING:
    from model_service.builders.base import BuilderContext


async def empty(_context: BuilderContext) -> BuilderResult:
    return BuilderResult({})


def application(builder: Any, adapter: Any, **options: Any) -> BuilderApplication:
    return BuilderApplication(spec(), builder, {"test": adapter}, **options)


class Adapter:
    def __init__(self, access: Any) -> None:
        self._access = access

    def create(self, _invocation: object, _model: object) -> Any:
        return self._access


class BuilderResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_deadline_includes_input_acquisition(self) -> None:
        cancelled = asyncio.Event()

        class BlockingAdapter:
            async def create(self, _invocation: object, _model: object) -> Any:
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        request = invocation(
            "test", deadline_at=datetime.now(UTC) + timedelta(milliseconds=20)
        )
        with self.assertRaises(BuilderFailure) as raised:
            await asyncio.wait_for(
                application(FunctionBuilder(empty), BlockingAdapter()).run(
                    request, SPATIAL_MODEL
                ),
                timeout=1,
            )
        self.assertEqual(raised.exception.code, "builder_deadline_exceeded")
        self.assertFalse(raised.exception.retryable)
        self.assertTrue(cancelled.is_set())

    async def test_cleanup_failure_preserves_primary_failure(self) -> None:
        class Access:
            async def aclose(self) -> None:
                raise OSError("cleanup failure")

        for primary in (
            InvalidInvocation("invalid configuration"),
            BuilderFailure(code="permanent", safe_message="cannot", retryable=False),
        ):

            async def build(_context: BuilderContext, primary: Any = primary) -> Any:
                raise primary

            with (
                self.subTest(failure=type(primary).__name__),
                self.assertRaises(type(primary)) as raised,
            ):
                await application(FunctionBuilder(build), Adapter(Access())).run(
                    invocation("test"), SPATIAL_MODEL
                )
            self.assertIs(raised.exception, primary)

    async def test_repeated_cancellation_waits_for_cleanup(self) -> None:
        started = asyncio.Event()
        closing = asyncio.Event()
        release_close = asyncio.Event()
        closed = asyncio.Event()

        async def build(_context: BuilderContext) -> BuilderResult:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError

        class Access:
            async def aclose(self) -> None:
                closing.set()
                await release_close.wait()
                closed.set()

        task = asyncio.create_task(
            application(FunctionBuilder(build), Adapter(Access())).run(
                invocation("test"), SPATIAL_MODEL
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        await asyncio.wait_for(closing.wait(), timeout=1)
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        release_close.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        self.assertTrue(closed.is_set())

    async def test_cleanup_is_bounded(self) -> None:
        close_cancelled = asyncio.Event()

        class Access:
            async def aclose(self) -> None:
                try:
                    await asyncio.Event().wait()
                finally:
                    close_cancelled.set()

        with self.assertRaises(BuilderFailure) as raised:
            await asyncio.wait_for(
                application(
                    FunctionBuilder(empty),
                    Adapter(Access()),
                    cleanup_timeout_seconds=0.02,
                ).run(invocation("test"), SPATIAL_MODEL),
                timeout=1,
            )
        self.assertEqual(raised.exception.code, "builder_internal_error")
        await asyncio.wait_for(close_cancelled.wait(), timeout=1)

    async def test_resource_self_cancellation_is_a_cleanup_failure(self) -> None:
        class Access:
            async def aclose(self) -> None:
                raise asyncio.CancelledError

        with self.assertRaises(BuilderFailure) as raised:
            await application(FunctionBuilder(empty), Adapter(Access())).run(
                invocation("test"), SPATIAL_MODEL
            )
        self.assertEqual(raised.exception.code, "builder_internal_error")

        async def build(_context: BuilderContext) -> BuilderResult:
            raise InvalidInvocation("primary failure")

        with self.assertRaisesRegex(InvalidInvocation, "primary failure"):
            await application(FunctionBuilder(build), Adapter(Access())).run(
                invocation("test"), SPATIAL_MODEL
            )


if __name__ == "__main__":
    unittest.main()
