"""Forward raw model requests from the public origin to configured Ray Serve origins.

This path preserves binary request/response payloads and selected headers. It bypasses
workflow invocation validation and durable scheduling.
"""

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
    """Carry the original model path, query string, binary payload, and allowed inference headers.

    Attributes:
        path: Requested model route following the public origin.
        query: Original URL query string, forwarded unchanged.
        body: Raw binary inference payload, forwarded unchanged.
        headers: Supported inference request headers retained by the public route.
    """

    path: str
    query: str
    body: bytes
    headers: dict[str, str]


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    """Carry the upstream HTTP status, payload bytes, and response headers safe to forward.

    Attributes:
        status_code: HTTP status of the upstream response.
        content: Raw response payload bytes.
        headers: Permitted HTTP request or response metadata.
    """

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
        """Send a raw request upstream and return its status, bytes, and permitted headers.

        Args:
            request: Raw model path, query, payload bytes, and permitted inference
                headers.

        Returns:
            InferenceResponse: Upstream status, binary response bytes, and permitted
                HTTP headers.

        Raises:
            InferenceProxyUnavailableError: The upstream raw inference request fails.
        """
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
        """Normalize configured origins and retain the caller-owned client and inference timeout.

        Args:
            client: Caller-owned HTTP client; this adapter does not close it.
            origin: Upstream service HTTP origin, normalized without a trailing slash.
            model_origins: Optional exact model-prefix to upstream HTTP origin
                overrides.
            timeout: Request or scenario time budget in seconds.
        """
        self._client = client
        self._origin = origin.rstrip("/")
        self._model_origins = {
            model: url.rstrip("/") for model, url in (model_origins or {}).items()
        }
        self._timeout = timeout

    async def forward(self, request: InferenceRequest) -> InferenceResponse:
        """POST unchanged payload bytes and query parameters, preserving upstream status while filtering transport headers.

        Args:
            request: Raw model path, query, payload bytes, and permitted inference
                headers.

        Returns:
            InferenceResponse: Upstream status, binary response bytes, and permitted
                HTTP headers.

        Raises:
            InferenceProxyUnavailableError: The upstream raw inference request fails.
        """
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
    """Create a catch-all model POST route for registration after the explicit workflow and execution routes.

    Args:
        proxy: Optional raw model request forwarding boundary.

    Returns:
        APIRouter: Catch-all raw model POST router for inclusion after explicit public
            routes.
    """
    router = APIRouter()

    @router.post("/{model_path:path}", include_in_schema=False)
    async def forward_model_inference(
        request: Request,
        model_path: str,
    ) -> Response:
        """Read the incoming binary request, forward permitted headers, and return the upstream response or a safe 503.

        Args:
            request: Incoming HTTP request containing the binary body, query string, and
                headers.
            model_path: Raw inference path following the public origin.

        Returns:
            Response: Upstream HTTP status and binary body with transport headers
                filtered.

        Raises:
            HTTPException: No proxy is configured or upstream transport fails, producing
                a safe HTTP 503.
        """
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
    """Keep only the supported inference request headers, excluding unrelated or credential-bearing headers.

    Args:
        headers: HTTP metadata to filter, send, or include in an error response.

    Returns:
        dict[str, str]: Only Accept, Content-Type, X-Output-Dtype, and X-Pool-Tokens
            headers supplied by the request.
    """
    return {
        name: value
        for name, value in headers.items()
        if name.lower() in _REQUEST_HEADERS
    }


def _unavailable(message: str) -> HTTPException:
    """Build the safe retryable 503 response for an unavailable inference proxy.

    Args:
        message: Safe user-facing failure or cancellation text.

    Returns:
        HTTPException: Structured safe HTTP error ready to raise at the route boundary.
    """
    return HTTPException(
        status_code=503,
        detail={
            "code": InferenceErrorCode.MODEL_INFERENCE_UNAVAILABLE,
            "message": message,
            "retryable": True,
        },
    )
