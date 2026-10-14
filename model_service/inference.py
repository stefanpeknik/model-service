"""Raw model requests: the transient proxy and the target for scheduled requests.

Every model takes the same raw input, an LZ4 frame holding one HWC uint8 RGB
tile of the model's `tile_size`. Scheduled requests are checked against that
before admission and before each attempt; the proxy forwards unchecked.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

import httpx
import lz4.frame
from fastapi import APIRouter, HTTPException, Request, Response

from execution_scheduler.inference import (
    IncompatibleInputError,
    ModelResponse,
    ModelUnavailableError,
    ResultTooLargeError,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from builders.contracts import ModelBinding
    from model_service.catalog import ModelCatalog


_REQUEST_HEADERS = frozenset(
    {"accept", "content-type", "x-output-dtype", "x-pool-tokens"}
)
_RESPONSE_EXCLUDED_HEADERS = frozenset(
    {"connection", "content-length", "keep-alive", "transfer-encoding", "upgrade"}
)


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """Carry the original model path, query string, binary payload, and allowed inference headers."""

    path: str
    query: str
    body: bytes
    headers: dict[str, str]


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    """Carry the upstream HTTP status, payload bytes, and response headers safe to forward."""

    status_code: int
    content: bytes
    headers: dict[str, str]


class InferenceProxyUnavailableError(RuntimeError):
    """Signal an upstream request failure without exposing HTTP client internals."""


class InferenceErrorCode(StrEnum):
    """Name the public failure for unavailable raw model inference."""

    MODEL_INFERENCE_UNAVAILABLE = "model_inference_unavailable"


class InferenceProxy(Protocol):
    """Define forwarding of raw model requests independently of the HTTP client implementation."""

    async def forward(self, request: InferenceRequest) -> InferenceResponse:
        """Send a raw request upstream and return its status, bytes, and permitted headers."""
        ...


class HttpInferenceProxy:
    """Forward inference to a default origin or a model-prefix-specific origin using a shared client."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        origin: str,
        model_origins: Mapping[str, str] | None = None,
        timeout: float = 900,
    ) -> None:
        """Normalize configured origins and retain the caller-owned client and inference timeout."""
        self._client = client
        self._origin = origin.rstrip("/")
        self._model_origins = {
            model: url.rstrip("/") for model, url in (model_origins or {}).items()
        }
        self._timeout = timeout

    async def forward(
        self, request: InferenceRequest, *, max_content: int | None = None
    ) -> InferenceResponse:
        """POST unchanged payload bytes and query parameters, preserving upstream status while filtering transport headers.

        Raises ResultTooLargeError once the response exceeds `max_content` bytes.
        """
        path = request.path.lstrip("/")
        origin = self._model_origins.get(path.partition("/")[0], self._origin)
        try:
            async with self._client.stream(
                "POST",
                f"{origin}/{path}",
                params=request.query,
                content=request.body,
                headers=request.headers,
                timeout=self._timeout,
                follow_redirects=True,
            ) as response:
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    content += chunk
                    if max_content is not None and len(content) > max_content:
                        raise ResultTooLargeError
        except httpx.RequestError as error:
            raise InferenceProxyUnavailableError(
                "model inference is temporarily unavailable"
            ) from error
        return InferenceResponse(
            status_code=response.status_code,
            content=bytes(content),
            headers={
                name: value
                for name, value in response.headers.items()
                if name.lower() not in _RESPONSE_EXCLUDED_HEADERS
            },
        )


def check_tile(model: ModelBinding, payload: bytes) -> None:
    """Require one LZ4 frame holding exactly a `tile_size`² RGB uint8 tile.

    Decompression stops one byte past the expected size, so a small payload
    cannot expand into unbounded memory.
    """
    expected = model.tile_size * model.tile_size * 3
    decompressor = lz4.frame.LZ4FrameDecompressor()
    try:
        tile = decompressor.decompress(payload, max_length=expected + 1)
    except RuntimeError as error:
        raise IncompatibleInputError("payload is not an LZ4 frame") from error
    if len(tile) != expected or not decompressor.eof or decompressor.unused_data:
        raise IncompatibleInputError(
            f"model {model.id!r} takes one LZ4-compressed {model.tile_size}x"
            f"{model.tile_size}x3 uint8 tile"
        )


class ModelInference:
    """Scheduled raw requests against catalog models through the inference proxy."""

    def __init__(
        self,
        catalog: ModelCatalog,
        proxy: HttpInferenceProxy,
        *,
        max_result_bytes: int,
    ) -> None:
        self._catalog = catalog
        self._proxy = proxy
        self._max_result_bytes = max_result_bytes

    def check(self, model_id: str, payload: bytes) -> None:
        check_tile(self._catalog.get(model_id), payload)

    async def call(
        self, model_id: str, payload: bytes, headers: Mapping[str, str]
    ) -> ModelResponse:
        try:
            response = await self._proxy.forward(
                InferenceRequest(
                    path=f"{model_id}/", query="", body=payload, headers=dict(headers)
                ),
                max_content=self._max_result_bytes,
            )
        except InferenceProxyUnavailableError as error:
            raise ModelUnavailableError(str(error)) from error
        return ModelResponse(
            response.status_code,
            response.content,
            {name.lower(): value for name, value in response.headers.items()},
        )


def create_inference_router(proxy: InferenceProxy | None) -> APIRouter:
    """Create a catch-all model POST route for registration after the explicit workflow and execution routes."""
    router = APIRouter()

    @router.post("/{model_path:path}", include_in_schema=False)
    async def forward_model_inference(
        request: Request,
        model_path: str,
    ) -> Response:
        """Read the incoming binary request, forward permitted headers, and return the upstream response or a safe 503."""
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
    """Keep only the supported inference request headers, excluding unrelated or credential-bearing headers."""
    return {
        name.lower(): value
        for name, value in headers.items()
        if name.lower() in _REQUEST_HEADERS
    }


def _unavailable(message: str) -> HTTPException:
    """Build the safe retryable 503 response for an unavailable inference proxy."""
    return HTTPException(
        status_code=503,
        detail={
            "code": InferenceErrorCode.MODEL_INFERENCE_UNAVAILABLE,
            "message": message,
            "retryable": True,
        },
    )
