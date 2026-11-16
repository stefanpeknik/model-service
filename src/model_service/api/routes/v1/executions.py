"""`/v1/executions`: status, lookup and cancellation.

Submitting work is `/v1/builders/.../schedule` in builders.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, HTTPException
from pydantic import Field

from model_service.http_common.auth import Access, Action
from model_service.http_common.errors import (
    ERRORS,
    ApiErrorCode,
    http_error,
    unavailable,
)
from model_service.scheduler.public import (
    ExecutionView,
    SchedulingUnavailableError,
    UnknownExecutionError,
)


if TYPE_CHECKING:
    from model_service.scheduler.protocols import Scheduler


MAX_LOOKUP_IDS = 1000


@dataclass(frozen=True, slots=True)
class ExecutionLookup:
    """A batch lookup of executions.

    Attributes:
        ids: The execution ids, at most `MAX_LOOKUP_IDS`.
    """

    ids: Annotated[list[str], Field(max_length=MAX_LOOKUP_IDS)]


def create_router(scheduler: Scheduler, access: Access) -> APIRouter:
    """Build the `/v1/executions` router: status, lookup and cancellation.

    Args:
        scheduler: Reads and cancels executions.
        access: Guards the routes.

    Returns:
        The router.
    """
    router = APIRouter(prefix="/v1/executions")

    @router.get(
        "/{execution_id}",
        responses=ERRORS,
        dependencies=[access.require(Action.READ_EXECUTION)],
    )
    async def get_execution(execution_id: str) -> ExecutionView:
        """Get an execution's status.

        Args:
            execution_id: The execution's id.

        Returns:
            The execution as callers see it.

        Raises:
            HTTPException: 404 for an unknown execution, 503 when the repository
                is unavailable.
        """
        try:
            execution = await scheduler.get(execution_id)
        except SchedulingUnavailableError as error:
            raise unavailable(error) from error
        except UnknownExecutionError as error:
            raise _unknown_execution() from error
        return execution

    @router.post(
        "/lookup",
        responses=ERRORS,
        dependencies=[access.require(Action.READ_EXECUTION)],
    )
    async def lookup_executions(body: ExecutionLookup) -> list[ExecutionView]:
        """Views of the listed executions that exist; unknown IDs are left out."""
        try:
            return await scheduler.get_many(body.ids)
        except SchedulingUnavailableError as error:
            raise unavailable(error) from error

    @router.post(
        "/{execution_id}/cancel",
        responses=ERRORS,
        dependencies=[access.require(Action.CANCEL_EXECUTION)],
    )
    async def cancel_execution(execution_id: str) -> ExecutionView:
        """Cancel an execution that has not finished.

        Args:
            execution_id: The execution's id.

        Returns:
            The execution after cancellation.

        Raises:
            HTTPException: 404 for an unknown execution, 503 when the repository
                is unavailable.
        """
        try:
            return await scheduler.cancel(execution_id)
        except SchedulingUnavailableError as error:
            raise unavailable(error) from error
        except UnknownExecutionError as error:
            raise _unknown_execution() from error

    return router


def _unknown_execution() -> HTTPException:
    """Build the 404 for an execution that does not exist.

    Returns:
        The HTTP error to raise.
    """
    return http_error(404, ApiErrorCode.UNKNOWN_EXECUTION, "execution does not exist")
