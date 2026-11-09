"""`/v1/executions`: status, lookup, stored inference results and cancellation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, HTTPException, Response
from pydantic import Field

from model_service.facade.auth import Access, Action
from model_service.facade.errors import ERRORS, ApiErrorCode, http_error, unavailable
from model_service.scheduler.contracts import (
    ExecutionView,
    ResultUnavailableError,
    SchedulingUnavailableError,
    UnknownExecutionError,
)


if TYPE_CHECKING:
    from model_service.facade.ports import Capacity, Scheduler


MAX_LOOKUP_IDS = 1000


@dataclass(frozen=True, slots=True)
class ExecutionLookup:
    ids: Annotated[list[str], Field(max_length=MAX_LOOKUP_IDS)]


def create_router(
    scheduler: Scheduler, capacity: Capacity, access: Access
) -> APIRouter:
    """Build the `/v1/executions` router: status, lookup, results and cancellation.

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
        "/lookup",
        responses=ERRORS,
        dependencies=[access.require(Action.READ_EXECUTION)],
    )
    async def lookup_executions(body: ExecutionLookup) -> list[ExecutionView]:
        """Views of the listed executions that exist; unknown IDs are left out."""
        try:
            executions = await scheduler.get_many(body.ids)
        except SchedulingUnavailableError as error:
            raise unavailable(error) from error
        return [await capacity.explain(execution) for execution in executions]

    @router.get(
        "/{execution_id}/result",
        responses={
            200: {"content": {"application/octet-stream": {}}},
            **ERRORS,
        },
        dependencies=[access.require(Action.READ_EXECUTION)],
    )
    async def get_inference_result(execution_id: str) -> Response:
        """The stored model response of a succeeded inference execution."""
        try:
            result, content = await scheduler.inference_result(execution_id)
        except SchedulingUnavailableError as error:
            raise unavailable(error) from error
        except UnknownExecutionError as error:
            raise _unknown_execution() from error
        except ResultUnavailableError as error:
            raise http_error(
                409,
                ApiErrorCode.RESULT_UNAVAILABLE,
                "execution has no inference response",
            ) from error
        return Response(
            content=content, media_type=result.media_type, headers=result.headers
        )

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
