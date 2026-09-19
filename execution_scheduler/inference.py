from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import httpx


if TYPE_CHECKING:
    from collections.abc import Mapping


_REQUEST_HEADERS = frozenset({"accept", "content-type", "x-output-dtype"})
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


class InferenceProxy(Protocol):
    async def forward(self, request: InferenceRequest) -> InferenceResponse: ...


class HttpInferenceProxy:
    def __init__(self, client: httpx.AsyncClient, *, origin: str) -> None:
        self._client = client
        self._origin = origin.rstrip("/")

    async def forward(self, request: InferenceRequest) -> InferenceResponse:
        try:
            response = await self._client.post(
                f"{self._origin}/{request.path.lstrip('/')}",
                params=request.query,
                content=request.body,
                headers=request.headers,
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


def inference_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() in _REQUEST_HEADERS
    }
