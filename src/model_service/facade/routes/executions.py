"""`/v1/executions`: status and cancellation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException

from model_service.facade.auth import Access, Action
from model_service.facade.errors import ERRORS, ApiErrorCode, http_error, unavailable
from model_service.scheduler.contracts import (
    ExecutionView,
    SchedulingUnavailableError,
    UnknownExecutionError,
)


if TYPE_CHECKING:
    from model_service.facade.ports import Capacity, Scheduler


def create_router(
    scheduler: Scheduler, capacity: Capacity, access: Access
) -> APIRouter:
    """Build the `/v1/executions` router: status and cancellation.

    Args:
        scheduler: Reads and cancels executions.
        capacity: Explains running executions waiting for Ray capacity.
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
        try:
            execution = await scheduler.get(execution_id)
        except SchedulingUnavailableError as error:
            raise unavailable(error) from error
        except UnknownExecutionError as error:
            raise _unknown_execution() from error
        return await capacity.explain(execution)

    @router.post(
        "/{execution_id}/cancel",
        responses=ERRORS,
        dependencies=[access.require(Action.CANCEL_EXECUTION)],
    )
    async def cancel_execution(execution_id: str) -> ExecutionView:
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
