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
    slide_id: str
    storage_path: Path
    supported: bool


@dataclass(frozen=True, slots=True)
class SlideLevel:
    index: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class SlideMetadata:
    identifier: str
    width: int
    height: int
    pixel_size_nm_x: float
    pixel_size_nm_y: float
    levels: tuple[SlideLevel, ...]


@dataclass(frozen=True, slots=True)
class Rectangle:
    x: int
    y: int
    width: int
    height: int


class ArtifactPublisher(Protocol):
    async def publish(
        self,
        gateway: HttpEmpaiaGateway,
        name: str,
        artifact: SharedArtifact,
    ) -> None: ...


class RectangleMaskWriter(Protocol):
    def write(
        self,
        destination: Path,
        slide: SlideMetadata,
        rectangle: Rectangle,
    ) -> None: ...


class HttpEmpaiaGateway:
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
        self._before_mutation = callback

    async def get_configuration(self) -> dict[str, JsonValue]:
        value = await self._request_json(
            "GET",
            f"/v3/{self._job()}/configuration",
        )
        return ConfigurationResponse.parse(value).merged()

    async def materialize_input(self, name: str, destination: str) -> None:
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
        await self._request(
            "PUT",
            f"/v3/{self._job()}/finalize",
            mutation=True,
        )

    async def report_failure(self, message: str) -> None:
        await self._request(
            "PUT",
            f"/v3/{self._job()}/failure",
            body={"user_message": message[:255]},
            mutation=True,
        )

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def job_id(self) -> str:
        return self._job_id

    async def slide_metadata(self) -> SlideMetadata:
        return await self._slide_metadata()

    async def create_pixelmap(
        self,
        name: str,
        body: dict[str, object],
    ) -> str:
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
        await self._request(
            "PUT",
            f"/v3/{self._job()}/pixelmaps/{quote(pixelmap_id, safe='')}"
            f"/level/{level}/position/{tile_x}/{tile_y}/data",
            content=data,
            mutation=True,
        )

    async def _direct_access(self, name: str) -> DirectAccess | None:
        path = f"/v3/rationai/{self._job()}/inputs/{quote(name, safe='')}/direct-access"
        response = await self._send("GET", path, allow_not_found=True)
        if response.status_code == 404:
            return None
        direct = DirectAccessResponse.parse(_response_json(response))
        return DirectAccess(
            slide_id=direct.slide_id,
            storage_path=Path(direct.storage_path),
            supported=direct.direct_access_supported,
        )

    async def _slide_metadata(self) -> SlideMetadata:
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
        return quote(self._job_id, safe="")


def create_http_gateway(context: EmpaiaJobContext) -> HttpEmpaiaGateway:
    return HttpEmpaiaGateway(
        context.api_url,
        context.job_id,
        context.credential,
    )


def _primitive_output(name: str, job_id: str, value: JsonValue) -> dict[str, object]:
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
    rectangle = RectangleResponse.parse(value)
    return Rectangle(
        x=math.floor(rectangle.upper_left[0]),
        y=math.floor(rectangle.upper_left[1]),
        width=math.ceil(rectangle.width),
        height=math.ceil(rectangle.height),
    )


def _response_json(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError as error:
        raise _invalid("EMPAIA AppService returned an invalid response") from error


def _invalid(message: str) -> EmpaiaGatewayError:
    return EmpaiaGatewayError(message, retryable=False)
