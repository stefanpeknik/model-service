"""Implement AppService HTTP access and classify safe integration failures.

Input acquisition uses direct shared access when available or rectangle-mask
materialization. Output publication invokes a durable guard before writes and tracks
uncertain mutation outcomes.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from aiohttp import ClientError, ClientResponseError
from rationai.empaia.app_service import AppServiceClient, AppServiceDecodeError

from model_service.builders.contracts import InlineOutput, SharedArtifact
from model_service.providers.empaia.gateway import EmpaiaGatewayError, EmpaiaJobContext
from model_service.providers.empaia.materialization import (
    link_direct_input,
    write_rectangle_input,
)
from model_service.providers.empaia.responses import (
    ConfigurationResponse,
    DirectAccessResponse,
    PixelmapResponse,
    RectangleResponse,
    SlideResponse,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable

    from model_service.builders.contracts import BuilderOutput, JsonValue
    from model_service.providers.empaia.gateway import MutationGuard


@dataclass(frozen=True, slots=True)
class DirectAccess:
    """Describe a shared input file and the provider's direct-access support flag."""

    storage_path: Path
    supported: bool


@dataclass(frozen=True, slots=True)
class SlideLevel:
    """Describe the index and pixel dimensions of one slide pyramid level."""

    index: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class SlideMetadata:
    """Carry slide identity, base dimensions, nanometer pixel sizes, and pyramid levels."""

    identifier: str
    width: int
    height: int
    pixel_size_nm_x: float
    pixel_size_nm_y: float
    levels: tuple[SlideLevel, ...]


@dataclass(frozen=True, slots=True)
class Rectangle:
    """Carry integer slide-space coordinates and dimensions for a region of interest."""

    x: int
    y: int
    width: int
    height: int


class ArtifactPublisher(Protocol):
    """Define conversion and publication of a completed shared artifact into external job outputs."""

    async def publish(
        self,
        gateway: HttpEmpaiaGateway,
        name: str,
        artifact: SharedArtifact,
    ) -> None:
        """Publish one named artifact using the gateway's guarded external-write operations."""
        ...


class RectangleMaskWriter(Protocol):
    """Define local rasterization of a slide-space rectangle into a mask file."""

    def write(
        self,
        destination: Path,
        slide: SlideMetadata,
        rectangle: Rectangle,
    ) -> None:
        """Write a region mask with geometry and physical resolution derived from the slide."""
        ...


class HttpEmpaiaGateway:
    """Access one EMPAIA job through authenticated HTTP and pluggable image adapters."""

    def __init__(
        self,
        api_url: str,
        job_id: str,
        credential: str,
        *,
        client: AppServiceClient | None = None,
        artifact_publisher: ArtifactPublisher | None = None,
        rectangle_writer: RectangleMaskWriter | None = None,
    ) -> None:
        """Require complete connection settings and select the SDK client, TIFF publisher, and rectangle writer. close() closes even an injected client."""
        if not api_url or not job_id or not credential:
            raise EmpaiaGatewayError(
                "EMPAIA connection settings are incomplete",
                retryable=False,
            )
        self._job_id = job_id
        self._client = client or AppServiceClient(api_url, job_id, credential)
        if artifact_publisher is None or rectangle_writer is None:
            from model_service.providers.empaia.image_io import (
                TiffPixelmapPublisher,
                VipsRectangleMaskWriter,
            )

        self._artifact_publisher = artifact_publisher or TiffPixelmapPublisher()
        self._rectangle_writer = rectangle_writer or VipsRectangleMaskWriter()

    def set_before_mutation(self, callback: MutationGuard) -> None:
        """Install the async durable-claim guard used immediately before each external mutation."""
        self._client.set_before_mutation(callback)

    async def get_configuration(self) -> dict[str, JsonValue]:
        """GET and validate job configuration, rejecting overlapping global/customer keys."""
        value = await self._call(self._client.get_configuration())
        return ConfigurationResponse.parse(value).merged()

    async def materialize_input(self, name: str, directory: str) -> str:
        """Link a directly accessible shared input or fetch slide/rectangle metadata and write a local mask."""
        direct = await self._direct_access(name)
        if direct is not None:
            return str(link_direct_input(direct, Path(directory)))
        target = Path(directory) / "mask.tiff"
        slide = await self._slide_metadata()
        rectangle = _rectangle(await self._call(self._client.get_input_data(name)))
        write_rectangle_input(self._rectangle_writer, target, slide, rectangle)
        return str(target)

    async def publish_output(self, name: str, output: BuilderOutput) -> None:
        """POST a supported inline primitive or delegate shared artifact publication to the image adapter."""
        if isinstance(output, InlineOutput):
            await self._call(
                self._client.post_output_data(
                    name, _primitive_output(name, self._job_id, output.value)
                ),
                mutation=True,
            )
            return
        if isinstance(output, SharedArtifact):
            await self._artifact_publisher.publish(self, name, output)
            return
        raise _invalid("EMPAIA output type is unsupported")

    async def finalize(self) -> None:
        """PUT the guarded successful job-finalization request."""
        await self._call(self._client.put_finalize(), mutation=True)

    async def report_failure(self, message: str) -> None:
        """PUT a guarded failure request with the safe user message truncated to 255 characters."""
        await self._call(self._client.put_failure(message[:255]), mutation=True)

    async def close(self) -> None:
        """Close the gateway's HTTP client and release its network resources."""
        await self._client.close()

    @property
    def job_id(self) -> str:
        """Expose the external job ID for publication metadata."""
        return self._job_id

    async def slide_metadata(self) -> SlideMetadata:
        """Fetch validated slide geometry for artifact publication."""
        return await self._slide_metadata()

    async def create_pixelmap(
        self,
        name: str,
        body: dict[str, object],
    ) -> str:
        """Create a named pixelmap through a guarded POST and return its validated ID."""
        value = await self._call(
            self._client.post_output_data(name, body), mutation=True
        )
        try:
            return PixelmapResponse.parse(value).id
        except EmpaiaGatewayError as error:
            raise EmpaiaGatewayError(
                error.safe_message, retryable=False, outcome_unknown=True
            ) from error

    async def put_pixelmap_tile(
        self,
        pixelmap_id: str,
        level: int,
        tile_x: int,
        tile_y: int,
        data: bytes,
    ) -> None:
        """Upload one tile; a PUT of the same tile is idempotent, so transient failures retry."""
        for attempt in range(_TILE_ATTEMPTS):
            try:
                await self._call(
                    self._client.put_tile(pixelmap_id, level, tile_x, tile_y, data),
                    mutation=True,
                )
                return
            except EmpaiaGatewayError as error:
                if not error.retryable or attempt + 1 == _TILE_ATTEMPTS:
                    raise
                await asyncio.sleep(_TILE_RETRY_SECONDS * 2**attempt)

    async def _direct_access(self, name: str) -> DirectAccess | None:
        """Query the extension endpoint; only a 404 permits rectangle fallback, while other failures remain errors."""
        value = await self._call(self._client.get_direct_access(name))
        if value is None:
            return None
        direct = DirectAccessResponse.parse(value)
        return DirectAccess(
            storage_path=Path(direct.storage_path),
            supported=direct.direct_access_supported,
        )

    async def _slide_metadata(self) -> SlideMetadata:
        """Fetch and decode the job's slide input metadata."""
        value = await self._call(self._client.get_input_data("slide"))
        return _slide_metadata(value)

    async def _call[T](self, operation: Awaitable[T], *, mutation: bool = False) -> T:
        """Translate SDK failures without exposing upstream details."""
        try:
            return await operation
        except ClientResponseError as error:
            if error.status == 429 or error.status >= 500:
                raise EmpaiaGatewayError(
                    "EMPAIA AppService is temporarily unavailable",
                    retryable=True,
                    outcome_unknown=mutation,
                ) from error
            if error.status in {401, 403}:
                raise _invalid("EMPAIA rejected the job credential") from error
            raise _invalid("EMPAIA rejected the application request") from error
        except (ClientError, TimeoutError) as error:
            raise EmpaiaGatewayError(
                "EMPAIA AppService is temporarily unavailable",
                retryable=True,
                outcome_unknown=mutation,
            ) from error
        except AppServiceDecodeError as error:
            raise EmpaiaGatewayError(
                "EMPAIA AppService returned an invalid response",
                retryable=False,
                outcome_unknown=mutation,
            ) from error


_TILE_ATTEMPTS = 4
_TILE_RETRY_SECONDS = 1.0


def create_http_gateway(context: EmpaiaJobContext) -> HttpEmpaiaGateway:
    """Construct the default HTTP gateway from protected job settings."""
    return HttpEmpaiaGateway(
        context.api_url,
        context.job_id,
        context.credential,
    )


def _primitive_output(name: str, job_id: str, value: JsonValue) -> dict[str, object]:
    """Encode a bool, integer, float, or nonempty string with EMPAIA output and creator metadata; reject other JSON values."""
    output_type: str
    if isinstance(value, bool):
        output_type = "bool"
    elif isinstance(value, int):
        output_type = "integer"
    elif isinstance(value, float):
        output_type = "float"
    elif isinstance(value, str) and value:
        output_type = "string"
    else:
        raise _invalid("EMPAIA inline output must be a non-empty primitive value")
    return {
        "name": name,
        "type": output_type,
        "value": value,
        "creator_id": job_id,
        "creator_type": "job",
    }


def _slide_metadata(value: object) -> SlideMetadata:
    """Convert validated provider slide data into local geometry values."""
    slide = SlideResponse.parse(value)
    return SlideMetadata(
        identifier=slide.id,
        width=slide.extent.x,
        height=slide.extent.y,
        pixel_size_nm_x=slide.pixel_size_nm.x,
        pixel_size_nm_y=slide.pixel_size_nm.y,
        levels=tuple(
            SlideLevel(index=index, width=level.extent.x, height=level.extent.y)
            for index, level in enumerate(slide.levels)
        ),
    )


def _rectangle(value: object) -> Rectangle:
    """Floor rectangle origins and ceil dimensions when converting provider geometry to integer pixels."""
    rectangle = RectangleResponse.parse(value)
    return Rectangle(
        x=math.floor(rectangle.upper_left[0]),
        y=math.floor(rectangle.upper_left[1]),
        width=math.ceil(rectangle.width),
        height=math.ceil(rectangle.height),
    )


def _invalid(message: str) -> EmpaiaGatewayError:
    """Construct a safe nonretryable gateway error for rejected or malformed integration data."""
    return EmpaiaGatewayError(message, retryable=False)
