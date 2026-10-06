from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from datetime import timedelta

    from execution_scheduler.domain import (
        Admission,
        AdmissionResult,
        ExecutionRecord,
        RenewDirective,
        Settlement,
        SubmissionIdentity,
        WorkClaim,
    )


class ExecutionRepository(Protocol):
    async def check_ready(self) -> None: ...

    async def maintain(self) -> None: ...

    async def find_submission(
        self,
        identity: SubmissionIdentity,
    ) -> ExecutionRecord | None: ...

    async def admit(self, admission: Admission) -> AdmissionResult: ...

    async def get(self, execution_id: str) -> ExecutionRecord: ...

    async def request_cancel(self, execution_id: str) -> ExecutionRecord: ...

    async def claim_due(
        self,
        owner: str,
        lease_duration: timedelta,
    ) -> WorkClaim | None: ...

    async def renew(
        self,
        execution_id: str,
        fencing_token: str,
        lease_duration: timedelta,
    ) -> RenewDirective: ...

    async def settle(
        self,
        execution_id: str,
        fencing_token: str,
        settlement: Settlement,
    ) -> ExecutionRecord: ...

    async def mark_finish_mutation_started(
        self,
        execution_id: str,
        fencing_token: str,
    ) -> None: ...


class ExecutionRepositoryUnavailableError(RuntimeError):
    pass
