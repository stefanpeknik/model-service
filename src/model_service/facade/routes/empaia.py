"""`/v1/providers/empaia`: schedule a builder for an EMPAIA job."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Response, status
from pydantic import Field

from model_service.facade.routes.common import ERRORS, Scheduler, submit
from model_service.scheduler.contracts import ScheduleRequest, ScheduleResult


@dataclass(frozen=True, slots=True)
class EmpaiaScheduleBody:
    model_id: Annotated[str, Field(min_length=1)]
    installation_id: str
    api_url: str
    job_id: str
    credential: str
    requested_timeout_seconds: int | None = None


def create_router(scheduler: Scheduler) -> APIRouter:
    router = APIRouter(prefix="/v1/providers/empaia")

    @router.post(
        "/builders/{builder_id}/schedule",
        status_code=status.HTTP_202_ACCEPTED,
        responses={200: {"model": ScheduleResult}, **ERRORS},
    )
    async def schedule_empaia(
        builder_id: str, body: EmpaiaScheduleBody, response: Response
    ) -> ScheduleResult:
        return await submit(
            scheduler,
            ScheduleRequest(
                builder_id=builder_id,
                model_id=body.model_id,
                provider="empaia",
                body={
                    "installation_id": body.installation_id,
                    "api_url": body.api_url,
                    "job_id": body.job_id,
                    "credential": body.credential,
                },
                requested_timeout_seconds=body.requested_timeout_seconds,
            ),
            response,
        )

    return router
