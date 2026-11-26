"""What the execution repository needs from a storage: records in transactions.

A storage only reads and writes; the repository decides. Records it returns
carry no provider context: that stays sealed in storage until a claim opens it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Collection
    from contextlib import AbstractAsyncContextManager
    from datetime import datetime

    from model_service.control_plane.scheduler.internal_state import ExecutionRecord
    from model_service.control_plane.scheduler.public import SubmissionIdentity


class StorageUnit(ABC):
    """One transaction: its writes apply together when it ends without an error."""

    @abstractmethod
    async def now(self) -> datetime:
        """The storage's clock, which every worker shares for leases and deadlines."""

    @abstractmethod
    async def lock_claims(self) -> None:
        """Hold the claim lock until the unit ends, so claims never race each other."""

    @abstractmethod
    async def insert(self, execution: ExecutionRecord, sealed_context: bytes) -> bool:
        """Add a new execution unless its submission identity is taken.

        Args:
            execution: The new record.
            sealed_context: Its encrypted provider context.

        Returns:
            False when an execution with the same provider and key exists.
        """

    @abstractmethod
    async def find(self, identity: SubmissionIdentity) -> ExecutionRecord | None:
        """Return the execution admitted under a provider and key, whatever its fingerprint.

        Args:
            identity: Replay identity of the submission.

        Returns:
            The record, or None.
        """

    @abstractmethod
    async def get(self, execution_id: str, *, lock: bool = False) -> ExecutionRecord:
        """Return one execution.

        Args:
            execution_id: ID of the execution.
            lock: Whether to hold the record against other units until this one ends.

        Returns:
            The record.

        Raises:
            UnknownExecutionError: No execution has this ID.
        """

    @abstractmethod
    async def get_many(self, execution_ids: Collection[str]) -> list[ExecutionRecord]:
        """Return the executions that exist, in the order requested.

        Args:
            execution_ids: IDs to read; unknown ones are skipped.

        Returns:
            The records found.
        """

    @abstractmethod
    async def with_work(self) -> list[ExecutionRecord]:
        """Return every execution with pending work or a lease, without locking them."""

    @abstractmethod
    async def sealed_context(self, execution_id: str) -> bytes:
        """Return an execution's encrypted provider context.

        Args:
            execution_id: ID of the execution.

        Returns:
            The stored bytes.
        """

    @abstractmethod
    async def save(
        self, execution: ExecutionRecord, *, sealed_context: bytes | None = None
    ) -> None:
        """Write an execution's changing state.

        Args:
            execution: The new state; its provider context is ignored.
            sealed_context: New encrypted context; None keeps the stored one.
        """


class ExecutionStorage(ABC):
    """Where executions live."""

    @abstractmethod
    def unit(self) -> AbstractAsyncContextManager[StorageUnit]:
        """Open a transaction.

        Raises:
            ExecutionRepositoryUnavailableError: The storage is unavailable.
        """

    @abstractmethod
    async def check_ready(self) -> None:
        """Check that the storage answers.

        Raises:
            ExecutionRepositoryUnavailableError: The storage is unavailable.
        """

    async def initialize(self) -> None:  # noqa: B027
        """Prepare the storage before first use; nothing by default."""

    async def aclose(self) -> None:  # noqa: B027
        """Release the storage's resources; nothing by default."""


class ExecutionRepositoryUnavailableError(RuntimeError):
    """Hide driver-specific storage failures behind the storage's availability contract."""
