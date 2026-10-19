"""`/v1/models`: the model catalog and raw inference, run directly or scheduled.

A raw request is one LZ4-compressed RGB tile of the model's `tile_size`, with
optional `x-output-dtype` and `x-pool-tokens` headers. The model's response is
returned (or, when scheduled, stored) as the model produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Header, HTTPException, Request, Response, status

from model_service.builders.contracts import (
    InferenceContract,
    JsonValue,
    ModelBinding,
    UnknownModelError,
)
from model_service.facade.inference import inference_headers
from model_service.facade.routes.common import (
    ERRORS,
    ApiErrorCode,
    Scheduler,
    http_error,
    lookup_error,
    submit,
)
from model_service.scheduler.contracts import InferenceScheduleRequest, ScheduleResult
from model_service.scheduler.inference import (
    IncompatibleInputError,
    ModelUnavailableError,
    ResultTooLargeError,
)


if TYPE_CHECKING:
    from model_service.facade.catalog import ModelCatalog
    from model_service.scheduler.inference import InferenceTarget


@dataclass(frozen=True, slots=True)
class ModelView:
    id: str
    contract: InferenceContract
    parameters: dict[str, JsonValue]
    builders: list[str]


def model_view(model: ModelBinding) -> ModelView:
    from model_service.facade.catalog import ModelCatalog

    return ModelView(
        id=model.id,
        contract=model.contract,
        parameters=model.model_dump(mode="json", exclude={"id", "contract"}),
        builders=ModelCatalog.compatible_builders(model),
    )


_BINARY_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
        },
    }
}


def create_router(
    scheduler: Scheduler,
    catalog: ModelCatalog,
    inference: InferenceTarget | None,
    max_request_bytes: int,
) -> APIRouter:
    router = APIRouter(prefix="/v1/models")

    @router.get("")
    async def list_models() -> list[ModelView]:
        return [model_view(model) for model in catalog]

    @router.get("/{model_id}", responses=ERRORS)
    async def get_model(model_id: str) -> ModelView:
        try:
            return model_view(catalog.get(model_id))
        except UnknownModelError as error:
            raise lookup_error(error) from error

    @router.post(
        "/{model_id}/run",
        responses={200: {"content": {"application/octet-stream": {}}}, **ERRORS},
        openapi_extra=_BINARY_BODY,
    )
    async def run_inference(model_id: str, request: Request) -> Response:
        """Send one raw request to the model and return its response."""
        if inference is None:
            raise http_error(
                503,
                ApiErrorCode.MODEL_UNAVAILABLE,
                "model inference is unavailable",
                retryable=True,
            )
        payload = await _read_body(request, max_request_bytes)
        try:
            inference.check(model_id, payload)
            response = await inference.call(
                model_id, payload, inference_headers(request.headers)
            )
        except UnknownModelError as error:
            raise lookup_error(error) from error
        except IncompatibleInputError as error:
            raise http_error(422, ApiErrorCode.INVALID_REQUEST, str(error)) from error
        except ModelUnavailableError as error:
            raise http_error(
                503, ApiErrorCode.MODEL_UNAVAILABLE, str(error), retryable=True
            ) from error
        except ResultTooLargeError as error:
            raise http_error(
                502,
                ApiErrorCode.MODEL_UNAVAILABLE,
                "model response exceeds the result limit",
            ) from error
        return Response(
            status_code=response.status_code,
            content=response.content,
            headers={
                name: value
                for name, value in response.headers.items()
                if name == "content-type" or name.startswith("x-")
            },
        )

    @router.post(
        "/{model_id}/schedule",
        status_code=status.HTTP_202_ACCEPTED,
        responses={200: {"model": ScheduleResult}, **ERRORS},
        openapi_extra=_BINARY_BODY,
    )
    async def schedule_inference(
        model_id: str,
        request: Request,
        response: Response,
        idempotency_key: Annotated[str | None, Header(min_length=1)] = None,
        timeout_seconds: int | None = None,
    ) -> ScheduleResult:
        """Durably queue one raw model request (the body `run` takes)."""
        if idempotency_key is None:
            raise http_error(
                422, ApiErrorCode.INVALID_REQUEST, "Idempotency-Key header is required"
            )
        payload = await _read_body(request, max_request_bytes)
        return await submit(
            scheduler,
            InferenceScheduleRequest(
                model_id=model_id,
                payload=payload,
                headers={
                    name: value
                    for name, value in inference_headers(request.headers).items()
                    if name.startswith("x-")
                },
                idempotency_key=idempotency_key,
                requested_timeout_seconds=timeout_seconds,
            ),
            response,
        )

    return router


async def _read_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise _payload_too_large(limit)
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise _payload_too_large(limit)
    return bytes(body)


def _payload_too_large(limit: int) -> HTTPException:
    return http_error(
        413, ApiErrorCode.PAYLOAD_TOO_LARGE, f"request body exceeds {limit} bytes"
    )
