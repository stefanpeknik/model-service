"""Implement AppService HTTP access and classify safe integration failures.

Input acquisition uses direct shared access when available or rectangle-mask
materialization. Where the slide storage is not visible to model-service,
`EMPAIA_SLIDE_ACCESS=http` downloads slides through the App API's region route
instead. Output publication invokes a durable guard before writes and tracks
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
from model_service.providers.empaia.gateway import (
    EmpaiaGatewayError,
    EmpaiaJobContext,
    EmpaiaJobMode,
)
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
from model_service.providers.empaia.settings import EmpaiaSettings, SlideAccess


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


class SlideDownloader(Protocol):
    """Define download of one slide level into a local TIFF readable by OpenSlide."""

    async def download(
        self,
        gateway: HttpEmpaiaGateway,
        slide: SlideMetadata,
        mpp: float,
        destination: Path,
    ) -> None:
        """Write the slide level closest to `mpp`, read through the gateway, to `destination`."""
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
        slide_access: SlideAccess = SlideAccess.DIRECT,
        slide_downloader: SlideDownloader | None = None,
    ) -> None:
        """Require complete connection settings and select the SDK client, TIFF publisher, and rectangle writer. close() closes even an injected client."""
        if not api_url or not job_id or not credential:
            raise EmpaiaGatewayError(
                "EMPAIA connection settings are incomplete",
                retryable=False,
            )
        self._job_id = job_id
        self._client = client or AppServiceClient(api_url, job_id, credential)
        if (
            artifact_publisher is None
            or rectangle_writer is None
            or slide_downloader is None
        ):
            from model_service.providers.empaia.image_io import (
                TiffPixelmapPublisher,
                TiffSlideDownloader,
                VipsRectangleMaskWriter,
            )

        self._artifact_publisher = artifact_publisher or TiffPixelmapPublisher()
        self._rectangle_writer = rectangle_writer or VipsRectangleMaskWriter()
        self._slide_access = slide_access
        self._slide_downloader = slide_downloader or TiffSlideDownloader()

    def set_before_mutation(self, callback: MutationGuard) -> None:
        """Install the async durable-claim guard used immediately before each external mutation."""
        self._client.set_before_mutation(callback)

    async def get_configuration(self) -> dict[str, JsonValue]:
        """GET and validate job configuration, rejecting overlapping global/customer keys."""
        value = await self._call(self._client.get_configuration())
        return ConfigurationResponse.parse(value).merged()

    async def get_mode(self) -> EmpaiaJobMode:
        value = await self._call(self._client.get_mode())
        try:
            return EmpaiaJobMode(value)
        except ValueError as error:
            raise EmpaiaGatewayError.invalid("EMPAIA job mode is unknown") from error

    async def has_input(self, name: str) -> bool:
        """AppService answers 404 for an input key the job does not have."""
        try:
            await self._call(self._client.get_input_data(name))
        except EmpaiaGatewayError as error:
            cause = error.__cause__
            if isinstance(cause, ClientResponseError) and cause.status == 404:
                return False
            raise
        return True

    async def materialize_input(
        self, name: str, directory: str, *, mpp: float | None = None
    ) -> str:
        """Link or download a slide input, or fetch rectangle metadata and write a local mask."""
        if self._slide_access is SlideAccess.DIRECT:
            direct = await self._direct_access(name)
            if direct is not None:
                return str(link_direct_input(direct, Path(directory)))
        value = await self._call(self._client.get_input_data(name))
        if self._slide_access is SlideAccess.HTTP and _is_slide(value):
            if mpp is None:
                raise EmpaiaGatewayError.invalid(
                    "EMPAIA slide download needs the model resolution"
                )
            target = Path(directory) / "slide.tiff"
            await self._slide_downloader.download(
                self, _slide_metadata(value), mpp, target
            )
            return str(target)
        target = Path(directory) / "mask.tiff"
        slide = await self._slide_metadata()
        write_rectangle_input(self._rectangle_writer, target, slide, _rectangle(value))
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
        raise EmpaiaGatewayError.invalid("EMPAIA output type is unsupported")

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

    async def read_region(
        self,
        slide_id: str,
        level: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> bytes:
        """Read one slide region as an encoded image; transient failures retry."""
        for attempt in range(_READ_ATTEMPTS):
            try:
                return await self._call(
                    self._client.get_region_data(slide_id, level, x, y, width, height)
                )
            except EmpaiaGatewayError as error:
                if not error.retryable or attempt + 1 == _READ_ATTEMPTS:
                    raise
                await asyncio.sleep(_READ_RETRY_SECONDS * 2**attempt)
        raise AssertionError("unreachable")

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
                raise EmpaiaGatewayError.invalid(
                    "EMPAIA rejected the job credential"
                ) from error
            raise EmpaiaGatewayError.invalid(
                "EMPAIA rejected the application request"
            ) from error
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
_READ_ATTEMPTS = 4
_READ_RETRY_SECONDS = 1.0


def create_http_gateway(context: EmpaiaJobContext) -> HttpEmpaiaGateway:
    """Construct the default HTTP gateway from protected job settings and `EmpaiaSettings`."""
    settings = EmpaiaSettings()
    return HttpEmpaiaGateway(
        context.api_url,
        context.job_id,
        context.credential,
        client=AppServiceClient(
            context.api_url,
            context.job_id,
            context.credential,
            timeout=settings.app_service_timeout_seconds,
        ),
        slide_access=settings.slide_access,
    )


def _is_slide(value: object) -> bool:
    """Slide inputs carry pyramid levels; rectangles do not."""
    return isinstance(value, dict) and "levels" in value


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
        raise EmpaiaGatewayError.invalid(
            "EMPAIA inline output must be a non-empty primitive value"
        )
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
