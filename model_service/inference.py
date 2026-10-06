from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

import httpx
from fastapi import APIRouter, HTTPException, Request, Response


if TYPE_CHECKING:
    from collections.abc import Mapping


_REQUEST_HEADERS = frozenset(
    {"accept", "content-type", "x-output-dtype", "x-pool-tokens"}
)
_RESPONSE_EXCLUDED_HEADERS = frozenset(
    {"connection", "content-length", "keep-alive", "transfer-encoding", "upgrade"}
)


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    path: str
    query: str
    body: bytes
    headers: dict[str, str]


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    status_code: int
    content: bytes
    headers: dict[str, str]


class InferenceProxyUnavailableError(RuntimeError):
    pass


class InferenceErrorCode(StrEnum):
    MODEL_INFERENCE_UNAVAILABLE = "model_inference_unavailable"


class InferenceProxy(Protocol):
    async def forward(self, request: InferenceRequest) -> InferenceResponse: ...


class HttpInferenceProxy:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        origin: str,
        model_origins: Mapping[str, str] | None = None,
        timeout: float = 900,
    ) -> None:
        self._client = client
        self._origin = origin.rstrip("/")
        self._model_origins = {
            model: url.rstrip("/") for model, url in (model_origins or {}).items()
        }
        self._timeout = timeout

    async def forward(self, request: InferenceRequest) -> InferenceResponse:
        path = request.path.lstrip("/")
        origin = self._model_origins.get(path.partition("/")[0], self._origin)
        try:
            response = await self._client.post(
                f"{origin}/{path}",
                params=request.query,
                content=request.body,
                headers=request.headers,
                timeout=self._timeout,
                follow_redirects=True,
            )
        except httpx.RequestError as error:
            raise InferenceProxyUnavailableError(
                "model inference is temporarily unavailable"
            ) from error
        return InferenceResponse(
            status_code=response.status_code,
            content=response.content,
            headers={
                name: value
                for name, value in response.headers.items()
                if name.lower() not in _RESPONSE_EXCLUDED_HEADERS
            },
        )


def create_inference_router(proxy: InferenceProxy | None) -> APIRouter:
    router = APIRouter()

    @router.post("/{model_path:path}", include_in_schema=False)
    async def forward_model_inference(
        request: Request,
        model_path: str,
    ) -> Response:
        if proxy is None:
            raise _unavailable("model inference is unavailable")
        try:
            forwarded = await proxy.forward(
                InferenceRequest(
                    path=model_path,
                    query=request.url.query,
                    body=await request.body(),
                    headers=inference_headers(request.headers),
                )
            )
        except InferenceProxyUnavailableError as error:
            raise _unavailable(str(error)) from error
        return Response(
            status_code=forwarded.status_code,
            content=forwarded.content,
            headers=forwarded.headers,
        )

    return router


def inference_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() in _REQUEST_HEADERS
    }


def _unavailable(message: str) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={
            "code": InferenceErrorCode.MODEL_INFERENCE_UNAVAILABLE,
            "message": message,
            "retryable": True,
        },
    )
