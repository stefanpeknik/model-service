"""Raw model requests, run directly or scheduled.

Every model takes the same raw input, an LZ4 frame holding one HWC uint8 RGB
tile of the model's `tile_size`. Requests are checked against that before they
are sent: a direct run before forwarding, a scheduled one at admission and
before each attempt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
import lz4.frame

from model_service.scheduler.inference import (
    IncompatibleInputError,
    ModelResponse,
    ModelUnavailableError,
    ResultTooLargeError,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.builders.contracts import ModelBinding
    from model_service.facade.catalog import ModelCatalog


_REQUEST_HEADERS = frozenset({"x-output-dtype", "x-pool-tokens"})
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
        """Normalize configured origins and keep the caller-owned client and timeout.

        Args:
            client: Shared HTTP client, owned by the caller.
            origin: Default origin for models without their own.
            model_origins: Origins by model route prefix, overriding `origin`.
            timeout: Per-request timeout in seconds.
        """
        self._client = client
        self._origin = origin.rstrip("/")
        self._model_origins = {
            model: url.rstrip("/") for model, url in (model_origins or {}).items()
        }
        self._timeout = timeout

    async def forward(
        self, request: InferenceRequest, *, max_content: int | None = None
    ) -> InferenceResponse:
        """POST the payload and query unchanged to the model's origin.

        The upstream status is kept; transport headers are dropped from the response.

        Args:
            request: The model path, query, payload and allowed headers.
            max_content: Largest response body accepted, in bytes; None for no limit.

        Returns:
            The upstream status, body and forwardable headers.

        Raises:
            ResultTooLargeError: Once the response exceeds `max_content` bytes.
            InferenceProxyUnavailableError: If the upstream request fails.
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

    Args:
        model: The target model, whose `tile_size` sets the expected size.
        payload: The raw request body.

    Raises:
        IncompatibleInputError: If the payload is not an LZ4 frame, or does not
            decompress to exactly one tile with nothing left over.
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
        """Keep the catalog, proxy and response size limit.

        Args:
            catalog: Deployed models.
            proxy: Forwards requests to Ray Serve.
            max_result_bytes: Largest model response accepted, in bytes.
        """
        self._catalog = catalog
        self._proxy = proxy
        self._max_result_bytes = max_result_bytes

    def check(self, model_id: str, payload: bytes) -> None:
        """Check that the payload is a valid tile for the model.

        Args:
            model_id: The target model ID.
            payload: The raw request body.

        Raises:
            UnknownModelError: If the model is not in the catalog.
            IncompatibleInputError: If the payload is not one LZ4 tile of the model's size.
        """
        check_tile(self._catalog.get(model_id), payload)

    async def call(
        self, model_id: str, payload: bytes, headers: Mapping[str, str]
    ) -> ModelResponse:
        """Send one raw request to the model through the proxy.

        Args:
            model_id: The target model ID.
            payload: The raw request body.
            headers: Allowed inference headers to forward.

        Returns:
            The model's status, body and headers, with lower-cased header names.

        Raises:
            ModelUnavailableError: If the upstream request fails.
            ResultTooLargeError: If the response exceeds the result size limit.
        """
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


def inference_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Keep only the supported inference request headers.

    Unrelated and credential-bearing headers are dropped.

    Args:
        headers: The incoming request headers.

    Returns:
        `x-output-dtype` and `x-pool-tokens`, if present, with lower-cased names.
    """
    return {
        name.lower(): value
        for name, value in headers.items()
        if name.lower() in _REQUEST_HEADERS
    }
