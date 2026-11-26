"""The execution repository over in-memory storage, for deterministic local tests.

`MemoryStorage` keeps records in dictionaries behind one asyncio lock, so units
never interleave; an injected clock permits deadline and recovery tests without
real waiting. A unit's writes apply when it ends without an error.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from model_service.control_plane.scheduler.internal_state import RetryPolicy
from model_service.control_plane.scheduler.public import UnknownExecutionError
from model_service.control_plane.scheduler.store.protected_context import (
    ProtectedContextCipher,
)
from model_service.control_plane.scheduler.store.repository import ExecutionRepository
from model_service.control_plane.scheduler.store.storage import (
    ExecutionStorage,
    StorageUnit,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Mapping

    from model_service.contracts.invocation import ExecutionContext, Invocation
    from model_service.contracts.pipeline_http import PipelineRunRequest
    from model_service.contracts.result import RunResult
    from model_service.control_plane.scheduler.internal_state import ExecutionRecord
    from model_service.control_plane.scheduler.public import SubmissionIdentity
    from model_service.control_plane.scheduler.targets import CapacityLimits


TEST_CIPHER = ProtectedContextCipher.from_base64(
    base64.urlsafe_b64encode(bytes(range(32))).decode()
)


def utc_now() -> datetime:
    """Return the timezone-aware UTC wall clock used when no test clock is injected."""
    return datetime.now(UTC)


class MemoryStorage(ExecutionStorage):
    """Executions and their sealed contexts in dictionaries.

    Attributes:
        executions: Records by ID, without their contexts.
        contexts: Sealed provider contexts by execution ID.
    """

    def __init__(self, clock: Callable[[], datetime] = utc_now) -> None:
        """Start empty.

        Args:
            clock: The storage's clock.
        """
        self._clock = clock
        self._lock = asyncio.Lock()
        self.executions: dict[str, ExecutionRecord] = {}
        self.contexts: dict[str, bytes] = {}

    @asynccontextmanager
    async def unit(self) -> AsyncIterator[StorageUnit]:
        """Hold the lock for the unit and apply its writes if it ends without an error.

        Yields:
            The unit.
        """
        async with self._lock:
            unit = _MemoryUnit(self, self._clock)
            yield unit
            self.executions.update(unit.executions)
            self.contexts.update(unit.contexts)

    async def check_ready(self) -> None:
        """Wait for the lock, as a storage that answers would."""
        async with self._lock:
            return


class _MemoryUnit(StorageUnit):
    """Reads see the storage plus this unit's writes; the storage applies them on success."""

    def __init__(self, storage: MemoryStorage, clock: Callable[[], datetime]) -> None:
        """Start with no writes.

        Args:
            storage: The storage the unit reads and, on success, writes.
            clock: The storage's clock.
        """
        self._storage = storage
        self._clock = clock
        self.executions: dict[str, ExecutionRecord] = {}
        self.contexts: dict[str, bytes] = {}

    def _all(self) -> dict[str, ExecutionRecord]:
        """Stored records overlaid with this unit's writes."""
        return self._storage.executions | self.executions

    async def now(self) -> datetime:
        """Read the injected clock."""
        return self._clock()

    async def lock_claims(self) -> None:
        """Do nothing: units never interleave."""
        return None  # The storage lock already serializes every unit.

    async def insert(self, execution: ExecutionRecord, sealed_context: bytes) -> bool:
        """Add the execution unless its provider and key are taken."""
        if await self.find(execution.identity) is not None:
            return False
        await self.save(execution, sealed_context=sealed_context)
        return True

    async def find(self, identity: SubmissionIdentity) -> ExecutionRecord | None:
        """Return the execution with this provider and key, if any."""
        return next(
            (
                e
                for e in self._all().values()
                if (e.identity.provider, e.identity.key)
                == (identity.provider, identity.key)
            ),
            None,
        )

    async def get(self, execution_id: str, *, lock: bool = False) -> ExecutionRecord:
        """Return one execution; `lock` changes nothing, as units never interleave."""
        try:
            return self._all()[execution_id]
        except KeyError as error:
            raise UnknownExecutionError(execution_id) from error

    async def get_many(self, execution_ids: Collection[str]) -> list[ExecutionRecord]:
        """Return the executions that exist, in the order requested."""
        found = self._all()
        return [found[i] for i in execution_ids if i in found]

    async def with_work(self) -> list[ExecutionRecord]:
        """Return every execution with pending work or a lease."""
        return [
            e
            for e in self._all().values()
            if e.work_kind is not None or e.fencing_token is not None
        ]

    async def sealed_context(self, execution_id: str) -> bytes:
        """Return the execution's sealed context."""
        return (self._storage.contexts | self.contexts)[execution_id]

    async def save(
        self, execution: ExecutionRecord, *, sealed_context: bytes | None = None
    ) -> None:
        """Record the new state, without its context, for the storage to apply."""
        self.executions[execution.id] = replace(execution, protected_context={})
        if sealed_context is not None:
            self.contexts[execution.id] = sealed_context


class InMemoryExecutionRepository(ExecutionRepository):
    """The production repository over `MemoryStorage`, with test defaults.

    Attributes:
        storage: The storage, for tests that inspect what is at rest.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = utc_now,
        per_model_limit: int = 1,
        capacity_limits: CapacityLimits | None = None,
        processing_policy: RetryPolicy | None = None,
        finish_policy: RetryPolicy | None = None,
    ) -> None:
        """Build the repository over a new `MemoryStorage` with the test cipher.

        Args:
            clock: The storage's clock.
            per_model_limit: Slots of every capacity key, unless `capacity_limits` is given.
            capacity_limits: Slots per capacity key.
            processing_policy: Processing retry policy; 3 attempts by default.
            finish_policy: Provider finish retry policy; 3 attempts by default.
        """
        default_policy = RetryPolicy(
            max_attempts=3,
            initial_backoff=timedelta(seconds=5),
            maximum_backoff=timedelta(minutes=1),
        )
        self.storage = MemoryStorage(clock)
        super().__init__(
            self.storage,
            cipher=TEST_CIPHER,
            capacity_limits=capacity_limits or fixed_limit(per_model_limit),
            processing_policy=processing_policy or default_policy,
            finish_policy=finish_policy or default_policy,
        )


type Handler = Callable[
    [Invocation, ExecutionContext], RunResult | Awaitable[RunResult]
]


class FakeDispatch:
    """Run in-process handlers by pipeline ID and record what they received."""

    def __init__(self, handlers: Mapping[str, Handler]) -> None:
        self._handlers = dict(handlers)
        self.pipelines: list[PipelineRunRequest] = []

    async def run_pipeline(self, request: PipelineRunRequest) -> RunResult:
        self.pipelines.append(request)
        value = self._handlers[str(request.pipeline["id"])](
            request.invocation, request.execution_context
        )
        if inspect.isawaitable(value):
            return await value
        return value


def fixed_limit(limit: int) -> CapacityLimits:
    """The same capacity limit for every key."""
    if limit <= 0:
        raise ValueError("per-model limits must be positive")
    return lambda _key: limit
