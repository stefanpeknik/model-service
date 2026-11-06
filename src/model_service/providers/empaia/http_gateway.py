"""Implement AppService HTTP access and classify safe integration failures.

Slides are linked from the slide volume through the RationAI direct-access
route; rectangles become masks. Output publication invokes a durable guard before writes and tracks
uncertain mutation outcomes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from aiohttp import ClientError, ClientResponseError, ContentTypeError
from rationai.empaia import Client
from rationai.empaia.typing import primitives

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
    RectangleResponse,
    SlideResponse,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable

    from rationai.empaia.typing import SlideInfo
    from rationai.empaia.utils.scalar_output import PixelmapTileSet

    from model_service.builders.contracts import JsonValue
    from model_service.providers.empaia.gateway import (
        EmpaiaGatewayFactory,
        MutationGuard,
    )
    from model_service.providers.empaia.settings import EmpaiaSettings


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
        client: Client | None = None,
        rectangle_writer: RectangleMaskWriter | None = None,
    ) -> None:
        """Require complete connection settings and select the SDK client, slide downloader, and rectangle writer. close() closes even an injected client."""
        if not api_url or not job_id or not credential:
            raise EmpaiaGatewayError(
                "EMPAIA connection settings are incomplete",
                retryable=False,
            )
        self._job_id = job_id
        self._client = client or Client(api_url, job_id, credential)
        if rectangle_writer is None:
            from model_service.providers.empaia.image_io import VipsRectangleMaskWriter

        self._rectangle_writer = rectangle_writer or VipsRectangleMaskWriter()

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
        self,
        name: str,
        directory: str,
    ) -> str:
        """Link a slide input, or fetch rectangle metadata and write a local mask."""
        direct = await self._direct_access(name)
        if direct is not None:
            return str(link_direct_input(direct, Path(directory)))
        value = await self._call(self._client.get_input_data(name))
        if _is_slide(value):
            raise EmpaiaGatewayError.invalid(
                "EMPAIA slide is not on the slide volume model-service reads"
            )
        target = Path(directory) / "mask.tiff"
        slide = await self._slide_metadata()
        write_rectangle_input(self._rectangle_writer, target, slide, _rectangle(value))
        return str(target)

    async def post_primitive(
        self, name: str, output_type: str, value: JsonValue
    ) -> None:
        """POST the guarded primitive output."""
        await self._call(
            self._client.post_output(
                name, _primitive_output(name, self._job_id, output_type, value)
            ),
            mutation=True,
        )

    async def finalize(self) -> None:
        """PUT the guarded successful job-finalization request."""
        await self._call(self._client.put_finalize(), mutation=True)

    async def report_failure(self, message: str) -> None:
        """PUT a guarded failure request with the safe user message truncated to 255 characters."""
        await self._call(self._client.put_failure(message[:255]), mutation=True)

    async def close(self) -> None:
        """Close the gateway's HTTP client and release its network resources."""
        await self._client.close()

    async def slide_info(self) -> SlideInfo:
        """The slide with every level's downsample factor, as pixelmap tiles need."""
        return await self._call(self._client.get_input("slide"))

    async def publish_pixelmap(
        self, name: str, slide_id: str, tile_sets: tuple[PixelmapTileSet, ...]
    ) -> None:
        """Create the pixelmap and upload its levels; failures leave an unknown outcome."""
        try:
            await self._call(
                self._client.upload_pixelmap_pyramid(
                    key=name, name=name, wsi_id=slide_id, tile_sets=tile_sets
                ),
                mutation=True,
            )
        except (RuntimeError, OSError) as error:
            # The SDK gave up on a tile after its own retries, or could not write
            # its storage directly; the pixelmap exists.
            raise EmpaiaGatewayError(
                "EMPAIA pixelmap upload failed", retryable=False, outcome_unknown=True
            ) from error

    async def _direct_access(self, name: str) -> DirectAccess | None:
        """Query the extension endpoint; only a 404 permits rectangle fallback, while other failures remain errors."""
        info = await self._call(self._client.get_direct_access_info(name))
        # CDS answers the path under its own data directory (physical_path)
        # and relative to it (storage_path); model-service mounts the slide
        # volume where CDS does, so the absolute path resolves here too.
        if info is None or not info.path:
            return None
        return DirectAccess(
            storage_path=Path(info.path),
            supported=info.direct_access_supported,
        )

    async def _slide_metadata(self) -> SlideMetadata:
        """Fetch and decode the job's slide input metadata."""
        value = await self._call(self._client.get_input_data("slide"))
        return _slide_metadata(value)

    async def _call[T](self, operation: Awaitable[T], *, mutation: bool = False) -> T:
        """Translate SDK failures without exposing upstream details."""
        try:
            return await operation
        except ContentTypeError as error:
            # The App Service answered 2xx with something that is not JSON.
            raise EmpaiaGatewayError(
                "EMPAIA AppService returned an invalid response",
                retryable=False,
                outcome_unknown=mutation,
            ) from error
        except ClientResponseError as error:
            if error.status == 429 or error.status >= 500:
                raise EmpaiaGatewayError(
                    "EMPAIA AppService is temporarily unavailable",
                    retryable=True,
                    # A 429 refused the request; a 5xx may have applied it.
                    outcome_unknown=mutation and error.status != 429,
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
        except (ValueError, TypeError) as error:
            # Not JSON, or not the shape the SDK's models require.
            raise EmpaiaGatewayError(
                "EMPAIA AppService returned an invalid response",
                retryable=False,
                outcome_unknown=mutation,
            ) from error


def http_gateway_factory(settings: EmpaiaSettings) -> EmpaiaGatewayFactory:
    """The gateway factory of a deployment, configured once from its settings."""

    def create(context: EmpaiaJobContext) -> HttpEmpaiaGateway:
        return HttpEmpaiaGateway(
            context.api_url,
            context.job_id,
            context.credential,
            client=Client(
                context.api_url,
                context.job_id,
                context.credential,
                timeout=settings.app_service_timeout_seconds,
            ),
        )

    return create


def _is_slide(value: object) -> bool:
    """Slide inputs carry pyramid levels; rectangles do not."""
    return isinstance(value, dict) and "levels" in value


type _PostPrimitive = (
    primitives.PostBoolPrimitive
    | primitives.PostIntegerPrimitive
    | primitives.PostFloatPrimitive
    | primitives.PostStringPrimitive
)


_PRIMITIVES: dict[str, type[_PostPrimitive]] = {
    "bool": primitives.PostBoolPrimitive,
    "integer": primitives.PostIntegerPrimitive,
    "float": primitives.PostFloatPrimitive,
    "string": primitives.PostStringPrimitive,
}


def _primitive_output(
    name: str, job_id: str, output_type: str, value: JsonValue
) -> _PostPrimitive:
    """Encode a primitive of the given type with EMPAIA output and creator metadata."""
    kind = _PRIMITIVES.get(output_type)
    if kind is None:
        raise EmpaiaGatewayError.invalid(f"EMPAIA has no {output_type!r} primitive")
    try:
        return kind.model_validate(
            {
                "name": name,
                "type": output_type,
                "value": value,
                "creator_id": job_id,
                "creator_type": "job",
            }
        )
    except ValueError as error:
        raise EmpaiaGatewayError.invalid(
            "EMPAIA inline output is not a valid primitive"
        ) from error


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
