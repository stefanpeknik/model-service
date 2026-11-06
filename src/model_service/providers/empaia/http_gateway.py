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
    EmpaiaGateway,
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
        """Write a region mask with geometry and physical resolution derived from the slide.

        Args:
            destination: The mask file to write.
            slide: The slide the rectangle lies on.
            rectangle: The region, in slide level-0 pixels.

        Raises:
            EmpaiaGatewayError: If the rectangle misses the slide or the mask cannot be
                written.
        """
        ...


class HttpEmpaiaGateway(EmpaiaGateway):
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
        """Require complete connection settings and select the SDK client and rectangle writer.

        close() closes even an injected client.

        Args:
            api_url: The App Service URL of the job.
            job_id: The EMPAIA job ID.
            credential: The job's token.
            client: The SDK client; one is built from the connection settings if omitted.
            rectangle_writer: Rasterizes rectangle inputs; libvips by default.

        Raises:
            EmpaiaGatewayError: If `api_url`, `job_id` or `credential` is empty; not
                retryable.
        """
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
        """Install the async durable-claim guard used immediately before each external mutation.

        Args:
            callback: Awaited before each external write.
        """
        self._client.set_before_mutation(callback)

    async def get_configuration(self) -> dict[str, JsonValue]:
        """GET and validate job configuration, rejecting overlapping global/customer keys.

        Returns:
            The global and customer configuration, merged.

        Raises:
            EmpaiaGatewayError: Not retryable if the response is malformed or the
                sections share a key; otherwise as `_call` classifies a failed request.
        """
        value = await self._call(self._client.get_configuration())
        return ConfigurationResponse.parse(value).merged()

    async def get_mode(self) -> EmpaiaJobMode:
        """Fetch how the job was started.

        Returns:
            The job's mode.

        Raises:
            EmpaiaGatewayError: Not retryable if the mode is unknown; otherwise
                as `_call` classifies a failed request.
        """
        value = await self._call(self._client.get_mode())
        try:
            return EmpaiaJobMode(value)
        except ValueError as error:
            raise EmpaiaGatewayError.invalid("EMPAIA job mode is unknown") from error

    async def has_input(self, name: str) -> bool:
        """Check whether the job has the named input.

        AppService answers 404 for an input key the job does not have.

        Args:
            name: The EMPAIA input key.

        Returns:
            False on a 404, True if the input data was returned.

        Raises:
            EmpaiaGatewayError: For any failure other than a 404, as `_call` classifies a failed request.
        """
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
        """Link a slide input, or fetch rectangle metadata and write a local mask.

        Args:
            name: The EMPAIA input key.
            directory: The directory to put the file in.

        Returns:
            The path of the slide link or of `mask.tiff`.

        Raises:
            EmpaiaGatewayError: Not retryable if the slide is not on the slide volume
                or does not support direct access, or the rectangle or slide data is
                invalid; retryable if linking or writing the mask fails; otherwise
                as `_call` classifies a failed request.
        """
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
        """POST the guarded primitive output.

        Args:
            name: The EMPAIA output key.
            output_type: The EMPAIA primitive type, e.g. `integer`.
            value: The output's value.

        Raises:
            EmpaiaGatewayError: Not retryable if the type has no primitive or the value
                does not fit it; otherwise as `_call` classifies a failed request as a mutation.
        """
        await self._call(
            self._client.post_output(
                name, _primitive_output(name, self._job_id, output_type, value)
            ),
            mutation=True,
        )

    async def finalize(self) -> None:
        """PUT the guarded successful job-finalization request.

        Raises:
            EmpaiaGatewayError: As `_call` classifies a failed request as a mutation.
        """
        await self._call(self._client.put_finalize(), mutation=True)

    async def report_failure(self, message: str) -> None:
        """PUT a guarded failure request with the safe user message truncated to 255 characters.

        Args:
            message: The safe message to report.

        Raises:
            EmpaiaGatewayError: As `_call` classifies a failed request as a mutation.
        """
        await self._call(self._client.put_failure(message[:255]), mutation=True)

    async def close(self) -> None:
        """Close the gateway's HTTP client and release its network resources."""
        await self._client.close()

    async def slide_info(self) -> SlideInfo:
        """Fetch the slide with every level's downsample factor, as pixelmap tiles need.

        Returns:
            The job's `slide` input.

        Raises:
            EmpaiaGatewayError: As `_call` classifies a failed request.
        """
        return await self._call(self._client.get_input("slide"))

    async def publish_pixelmap(
        self, name: str, slide_id: str, tile_sets: tuple[PixelmapTileSet, ...]
    ) -> None:
        """Create the pixelmap and upload its levels; failures leave an unknown outcome.

        Args:
            name: The EMPAIA output key, also the pixelmap's name.
            slide_id: The slide the pixelmap belongs to.
            tile_sets: The pyramid levels to upload.

        Raises:
            EmpaiaGatewayError: Not retryable with an unknown outcome if the SDK gives
                up on a tile or cannot write its storage directly; otherwise as `_call` classifies a failed request
                as a mutation.
        """
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
        """Query the extension endpoint for the input's path on the slide volume.

        Only a 404 permits rectangle fallback, while other failures remain errors.

        Args:
            name: The EMPAIA input key.

        Returns:
            The path and whether direct access is supported, or None if the input has
            no direct-access path.

        Raises:
            EmpaiaGatewayError: As `_call` classifies a failed request.
        """
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
        """Fetch and decode the job's slide input metadata.

        Returns:
            The slide's identity, size, pixel size and levels.

        Raises:
            EmpaiaGatewayError: Not retryable if the slide data is invalid; otherwise
                as `_call` classifies a failed request.
        """
        value = await self._call(self._client.get_input_data("slide"))
        return _slide_metadata(value)

    async def _call[T](self, operation: Awaitable[T], *, mutation: bool = False) -> T:
        """Await an SDK call, translating failures without exposing upstream details.

        Args:
            operation: The SDK call.
            mutation: Whether the call writes; a failure that may have applied it is
                marked `outcome_unknown`.

        Returns:
            The call's result.

        Raises:
            EmpaiaGatewayError: Retryable on a 429, a 5xx, a connection error or a
                timeout; not retryable on a 401 or 403 (rejected credential), another
                HTTP error, or a response that is not JSON or not the expected shape.
                A failed mutation's outcome is unknown, except after a 429 or another
                4xx.
        """
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
    """Build the gateway factory of a deployment, configured once from its settings.

    Args:
        settings: Supplies the App Service timeout.

    Returns:
        A factory creating one `HttpEmpaiaGateway` per job.
    """

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
    """Check whether input data is a slide; slide inputs carry pyramid levels, rectangles do not.

    Args:
        value: The decoded input data.

    Returns:
        True for a slide.
    """
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
    """Encode a primitive of the given type with EMPAIA output and creator metadata.

    Args:
        name: The EMPAIA output key.
        job_id: The job, recorded as the creator.
        output_type: The EMPAIA primitive type, e.g. `integer`.
        value: The output's value.

    Returns:
        The primitive to post.

    Raises:
        EmpaiaGatewayError: Not retryable if EMPAIA has no such primitive type or
            the value does not fit it.
    """
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
    """Convert validated provider slide data into local geometry values.

    Args:
        value: The decoded slide input data.

    Returns:
        The slide's identity, size, pixel size and levels.

    Raises:
        EmpaiaGatewayError: Not retryable if the data is not a valid slide.
    """
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
    """Floor rectangle origins and ceil dimensions when converting provider geometry to integer pixels.

    Args:
        value: The decoded rectangle input data.

    Returns:
        The rectangle in integer pixels.

    Raises:
        EmpaiaGatewayError: Not retryable if the data is not a valid rectangle.
    """
    rectangle = RectangleResponse.parse(value)
    return Rectangle(
        x=math.floor(rectangle.upper_left[0]),
        y=math.floor(rectangle.upper_left[1]),
        width=math.ceil(rectangle.width),
        height=math.ceil(rectangle.height),
    )
