"""Implement AppService HTTP access and classify safe integration failures.

Input acquisition uses direct shared access when available or rectangle-mask
materialization. Output publication invokes a durable guard before writes and tracks
uncertain mutation outcomes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from urllib.parse import quote

import httpx

from integrations.empaia.gateway import EmpaiaGatewayError, EmpaiaJobContext
from integrations.empaia.materialization import link_direct_input, write_rectangle_input
from integrations.empaia.responses import (
    ConfigurationResponse,
    DirectAccessResponse,
    PixelmapResponse,
    RectangleResponse,
    SlideResponse,
)
from workflow_execution.contracts import InlineOutput, SharedArtifact


if TYPE_CHECKING:
    from integrations.empaia.gateway import MutationGuard
    from workflow_execution.contracts import JsonValue, WorkflowOutput


@dataclass(frozen=True, slots=True)
class DirectAccess:
    """Describe a shared input file and the provider's direct-access support flag.

    Attributes:
        storage_path: Provider shared file path or authorized storage locator.
        supported: Whether this provider input supports direct shared access.
    """

    storage_path: Path
    supported: bool


@dataclass(frozen=True, slots=True)
class SlideLevel:
    """Describe the index and pixel dimensions of one slide pyramid level.

    Attributes:
        index: Zero-based slide pyramid level index.
        width: Pixel width or positive rectangle width.
        height: Pixel height or positive rectangle height.
    """

    index: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class SlideMetadata:
    """Carry slide identity, base dimensions, nanometer pixel sizes, and pyramid levels.

    Attributes:
        identifier: External whole-slide image identifier.
        width: Pixel width or positive rectangle width.
        height: Pixel height or positive rectangle height.
        pixel_size_nm_x: Horizontal physical pixel size in nanometers.
        pixel_size_nm_y: Vertical physical pixel size in nanometers.
        levels: Available slide pyramid levels and their geometry.
    """

    identifier: str
    width: int
    height: int
    pixel_size_nm_x: float
    pixel_size_nm_y: float
    levels: tuple[SlideLevel, ...]


@dataclass(frozen=True, slots=True)
class Rectangle:
    """Carry integer slide-space coordinates and dimensions for a region of interest.

    Attributes:
        x: Horizontal origin in base slide pixels.
        y: Vertical origin in base slide pixels.
        width: Positive rectangle width in base slide pixels.
        height: Positive rectangle height in base slide pixels.
    """

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
        """Publish one named artifact using the gateway's guarded external-write operations.

        Args:
            gateway: Job-scoped provider boundary used for publication or cleanup.
            name: Declared workflow output name registered with AppService.
            artifact: Completed shared output reference to publish.

        Returns:
            None: No return value.
        """
        ...


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
            destination: Local path where input materialization or mask writing must
                produce a file.
            slide: Validated slide dimensions, pyramid levels, and nanometer pixel
                sizes.
            rectangle: Slide-space region origin and dimensions to rasterize.

        Returns:
            None: No return value.
        """
        ...


class HttpEmpaiaGateway:
    """Access one EMPAIA job through authenticated HTTP and pluggable image adapters."""

    def __init__(
        self,
        api_url: str,
        job_id: str,
        credential: str,
        *,
        client: httpx.AsyncClient | None = None,
        artifact_publisher: ArtifactPublisher | None = None,
        rectangle_writer: RectangleMaskWriter | None = None,
    ) -> None:
        """Require complete connection settings and select the HTTP client, TIFF publisher, and rectangle writer. close() closes even an injected client.

        Args:
            api_url: AppService origin for the external job.
            job_id: External EMPAIA job identifier.
            credential: Private bearer credential for the external job; never include it
                in public errors.
            client: Optional HTTP client; the gateway closes it, including when supplied
                by the caller.
            artifact_publisher: Artifact conversion/publication adapter; defaults to the
                TIFF pixelmap publisher.
            rectangle_writer: Optional mask rasterizer; defaults to the libvips writer.

        Raises:
            EmpaiaGatewayError: api_url, job_id, or credential is empty.
        """
        if not api_url or not job_id or not credential:
            raise EmpaiaGatewayError(
                "EMPAIA connection settings are incomplete",
                retryable=False,
            )
        self._job_id = job_id
        self._before_mutation: MutationGuard | None = None
        self._headers = {"Authorization": f"Bearer {credential}"}
        self._client = client or httpx.AsyncClient(base_url=api_url.rstrip("/"))
        if artifact_publisher is None or rectangle_writer is None:
            from integrations.empaia.image_io import (
                TiffPixelmapPublisher,
                VipsRectangleMaskWriter,
            )

        self._artifact_publisher = artifact_publisher or TiffPixelmapPublisher()
        self._rectangle_writer = rectangle_writer or VipsRectangleMaskWriter()

    def set_before_mutation(self, callback: MutationGuard) -> None:
        """Install the async durable-claim guard used immediately before each external mutation.

        Args:
            callback: Async callback invoked before an external provider mutation.

        Returns:
            None: No return value.
        """
        self._before_mutation = callback

    async def get_configuration(self) -> dict[str, JsonValue]:
        """GET and validate job configuration, rejecting overlapping global/customer keys.

        Returns:
            dict[str, JsonValue]: Global and customer configuration merged after
                checking that their keys are disjoint.

        Raises:
            EmpaiaGatewayError: AppService transport, authentication, response data, or
                image conversion fails.
        """
        value = await self._request_json(
            "GET",
            f"/v3/{self._job()}/configuration",
        )
        return ConfigurationResponse.parse(value).merged()

    async def materialize_input(self, name: str, destination: str) -> None:
        """Link a directly accessible shared input or fetch slide/rectangle metadata and write a local mask.

        Args:
            name: Declared workflow input name to resolve.
            destination: Local path where input materialization or mask writing must
                produce a file.

        Returns:
            None: No return value.

        Raises:
            EmpaiaGatewayError: AppService transport, authentication, response data, or
                image conversion fails.
        """
        target = Path(destination)
        direct = await self._direct_access(name)
        if direct is not None:
            link_direct_input(direct, target)
            return
        slide = await self._slide_metadata()
        rectangle = _rectangle(
            await self._request_json(
                "GET",
                f"/v3/{self._job()}/inputs/{quote(name, safe='')}",
            )
        )
        write_rectangle_input(self._rectangle_writer, target, slide, rectangle)

    async def publish_output(self, name: str, output: WorkflowOutput) -> None:
        """POST a supported inline primitive or delegate shared artifact publication to the image adapter.

        Args:
            name: Declared workflow output name registered with AppService.
            output: Inline JSON value or completed shared artifact to publish.

        Returns:
            None: No return value.

        Raises:
            EmpaiaGatewayError: AppService transport, authentication, response data, or
                image conversion fails.
        """
        if isinstance(output, InlineOutput):
            await self._request_json(
                "POST",
                f"/v3/{self._job()}/outputs/{quote(name, safe='')}",
                body=_primitive_output(name, self._job_id, output.value),
                mutation=True,
            )
            return
        if isinstance(output, SharedArtifact):
            await self._artifact_publisher.publish(self, name, output)
            return
        raise _invalid("EMPAIA output type is unsupported")

    async def finalize(self) -> None:
        """PUT the guarded successful job-finalization request.

        Returns:
            None: No return value.

        Raises:
            EmpaiaGatewayError: AppService transport, authentication, response data, or
                image conversion fails.
        """
        await self._request(
            "PUT",
            f"/v3/{self._job()}/finalize",
            mutation=True,
        )

    async def report_failure(self, message: str) -> None:
        """PUT a guarded failure request with the safe user message truncated to 255 characters.

        Args:
            message: Safe user-facing failure or cancellation text.

        Returns:
            None: No return value.

        Raises:
            EmpaiaGatewayError: AppService transport, authentication, response data, or
                image conversion fails.
        """
        await self._request(
            "PUT",
            f"/v3/{self._job()}/failure",
            body={"user_message": message[:255]},
            mutation=True,
        )

    async def close(self) -> None:
        """Close the gateway's HTTP client and release its network resources.

        Returns:
            None: No return value.
        """
        await self._client.aclose()

    @property
    def job_id(self) -> str:
        """Expose the external job ID for publication metadata.

        Returns:
            str: External job ID used in publication creator metadata.
        """
        return self._job_id

    async def slide_metadata(self) -> SlideMetadata:
        """Fetch validated slide geometry for artifact publication.

        Returns:
            SlideMetadata: Slide identity, geometry, pixel sizes, and available pyramid
                levels.

        Raises:
            EmpaiaGatewayError: AppService transport, authentication, response data, or
                image conversion fails.
        """
        return await self._slide_metadata()

    async def create_pixelmap(
        self,
        name: str,
        body: dict[str, object],
    ) -> str:
        """Create a named pixelmap through a guarded POST and return its validated ID.

        Args:
            name: Declared workflow output name registered with AppService.
            body: Typed or JSON request payload accepted by this boundary.

        Returns:
            str: Nonempty external pixelmap ID returned by the output creation response.

        Raises:
            EmpaiaGatewayError: AppService transport, authentication, response data, or
                image conversion fails.
        """
        value = await self._request_json(
            "POST",
            f"/v3/{self._job()}/outputs/{quote(name, safe='')}",
            body=body,
            mutation=True,
        )
        return PixelmapResponse.parse(value).id

    async def put_pixelmap_tile(
        self,
        pixelmap_id: str,
        level: int,
        tile_x: int,
        tile_y: int,
        data: bytes,
    ) -> None:
        """Upload binary tile values to a pixelmap level and position through a guarded PUT.

        Args:
            pixelmap_id: External output pixelmap identifier returned by AppService.
            level: Selected pyramid level index.
            tile_x: Horizontal pixelmap tile position.
            tile_y: Vertical pixelmap tile position.
            data: Raw float32 pixelmap tile bytes to upload.

        Returns:
            None: No return value.

        Raises:
            EmpaiaGatewayError: AppService transport, authentication, response data, or
                image conversion fails.
        """
        await self._request(
            "PUT",
            f"/v3/{self._job()}/pixelmaps/{quote(pixelmap_id, safe='')}"
            f"/level/{level}/position/{tile_x}/{tile_y}/data",
            content=data,
            mutation=True,
        )

    async def _direct_access(self, name: str) -> DirectAccess | None:
        """Query the extension endpoint; only a 404 permits rectangle fallback, while other failures remain errors.

        Args:
            name: Declared workflow input name to resolve.

        Returns:
            DirectAccess | None: Validated shared input metadata, or None only when the
                direct-access endpoint returns 404.

        Raises:
            EmpaiaGatewayError: AppService transport, authentication, response data, or
                image conversion fails.
        """
        path = f"/v3/rationai/{self._job()}/inputs/{quote(name, safe='')}/direct-access"
        response = await self._send("GET", path, allow_not_found=True)
        if response.status_code == 404:
            return None
        direct = DirectAccessResponse.parse(_response_json(response))
        return DirectAccess(
            storage_path=Path(direct.storage_path),
            supported=direct.direct_access_supported,
        )

    async def _slide_metadata(self) -> SlideMetadata:
        """Fetch and decode the job's slide input metadata.

        Returns:
            SlideMetadata: Slide identity, geometry, pixel sizes, and available pyramid
                levels.

        Raises:
            EmpaiaGatewayError: AppService transport, authentication, response data, or
                image conversion fails.
        """
        value = await self._request_json(
            "GET",
            f"/v3/{self._job()}/inputs/slide",
        )
        return _slide_metadata(value)

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, object] | None = None,
        mutation: bool = False,
    ) -> object:
        """Send a classified request and require a decodable JSON response.

        Args:
            method: HTTP method used by the request or test client.
            path: AppService endpoint path relative to the configured API origin.
            body: Typed or JSON request payload accepted by this boundary.
            mutation: Whether the request changes external job state and therefore
                requires mutation guarding.

        Returns:
            object: Decoded JSON body from a successful AppService response.

        Raises:
            EmpaiaGatewayError: AppService transport, authentication, response data, or
                image conversion fails.
        """
        response = await self._send(method, path, body=body, mutation=mutation)
        return _response_json(response)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, object] | None = None,
        content: bytes | None = None,
        mutation: bool = False,
    ) -> None:
        """Send a classified request whose response body is not needed.

        Args:
            method: HTTP method used by the request or test client.
            path: AppService endpoint path relative to the configured API origin.
            body: Typed or JSON request payload accepted by this boundary.
            content: Optional binary request payload sent unchanged.
            mutation: Whether the request changes external job state and therefore
                requires mutation guarding.

        Returns:
            None: No return value.

        Raises:
            EmpaiaGatewayError: AppService transport, authentication, response data, or
                image conversion fails.
        """
        await self._send(
            method,
            path,
            body=body,
            content=content,
            mutation=mutation,
        )

    async def _send(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, object] | None = None,
        content: bytes | None = None,
        mutation: bool = False,
        allow_not_found: bool = False,
    ) -> httpx.Response:
        """Apply the mutation guard, send authenticated HTTP, and classify transport/status failures.

        A failed mutation may have reached AppService, so transport errors, 429s, and
        server failures mark its outcome uncertain. The direct-access probe alone may
        accept a 404.

        Args:
            method: HTTP method used by the request or test client.
            path: AppService endpoint path relative to the configured API origin.
            body: Typed or JSON request payload accepted by this boundary.
            content: Optional binary request payload sent unchanged.
            mutation: Whether the request changes external job state and therefore
                requires mutation guarding.
            allow_not_found: Whether a 404 is an expected probe outcome instead of a
                gateway error.

        Returns:
            httpx.Response: Successful AppService response, or an allowed 404 response
                for optional direct access.

        Raises:
            EmpaiaGatewayError: AppService transport, authentication, response data, or
                image conversion fails.
            StaleWorkClaimError: The token, lease, deadline, or work kind no longer
                permits this operation.
        """
        if mutation and self._before_mutation is not None:
            await self._before_mutation()
        try:
            response = await self._client.request(
                method,
                path,
                headers=self._headers,
                json=body,
                content=content,
            )
        except httpx.TransportError as error:
            raise EmpaiaGatewayError(
                "EMPAIA AppService is temporarily unavailable",
                retryable=True,
                outcome_unknown=mutation,
            ) from error
        if response.status_code == 404 and allow_not_found:
            return response
        if response.is_success:
            return response
        if response.status_code == 429 or response.status_code >= 500:
            raise EmpaiaGatewayError(
                "EMPAIA AppService is temporarily unavailable",
                retryable=True,
                outcome_unknown=mutation,
            )
        if response.status_code in {401, 403}:
            raise _invalid("EMPAIA rejected the job credential")
        raise _invalid("EMPAIA rejected the application request")

    def _job(self) -> str:
        """URL-encode the complete job ID as one path component.

        Returns:
            str: URL-encoded job ID safe to embed as a single HTTP path component.
        """
        return quote(self._job_id, safe="")


def create_http_gateway(context: EmpaiaJobContext) -> HttpEmpaiaGateway:
    """Construct the default HTTP gateway from protected job settings.

    Args:
        context: External job origin/ID and protected bearer credential.

    Returns:
        HttpEmpaiaGateway: Configured default authenticated AppService HTTP gateway.

    Raises:
        EmpaiaGatewayError: The supplied context contains empty connection or
            authentication settings.
    """
    return HttpEmpaiaGateway(
        context.api_url,
        context.job_id,
        context.credential,
    )


def _primitive_output(name: str, job_id: str, value: JsonValue) -> dict[str, object]:
    """Encode a bool, integer, float, or nonempty string with EMPAIA output and creator metadata; reject other JSON values.

    Args:
        name: Declared workflow output name registered with AppService.
        job_id: External EMPAIA job identifier.
        value: Inline output value to encode as an EMPAIA primitive.

    Returns:
        dict[str, object]: EMPAIA primitive output fields with job creator metadata.

    Raises:
        EmpaiaGatewayError: value is not a boolean, integer, float, or nonempty string.
    """
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
    """Convert validated provider slide data into local geometry values.

    Args:
        value: AppService slide input response requiring geometry, physical pixel size,
            and levels.

    Returns:
        SlideMetadata: Slide identity, geometry, pixel sizes, and available pyramid
            levels.
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
        value: AppService rectangle response requiring finite origin coordinates and
            positive dimensions.

    Returns:
        Rectangle: Integer slide-space region coordinates and dimensions.
    """
    rectangle = RectangleResponse.parse(value)
    return Rectangle(
        x=math.floor(rectangle.upper_left[0]),
        y=math.floor(rectangle.upper_left[1]),
        width=math.ceil(rectangle.width),
        height=math.ceil(rectangle.height),
    )


def _response_json(response: httpx.Response) -> object:
    """Decode response JSON or raise a safe nonretryable response error.

    Args:
        response: Successful AppService response whose body must decode as JSON.

    Returns:
        object: Decoded AppService response JSON.

    Raises:
        EmpaiaGatewayError: The AppService response body cannot be decoded as JSON.
    """
    try:
        return response.json()
    except ValueError as error:
        raise _invalid("EMPAIA AppService returned an invalid response") from error


def _invalid(message: str) -> EmpaiaGatewayError:
    """Construct a safe nonretryable gateway error for rejected or malformed integration data.

    Args:
        message: Safe user-facing failure or cancellation text.

    Returns:
        EmpaiaGatewayError: Safe integration error carrying retry and uncertain-outcome
            metadata.
    """
    return EmpaiaGatewayError(message, retryable=False)
