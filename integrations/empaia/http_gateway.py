from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import quote

import httpx

from integrations.empaia.gateway import EmpaiaGatewayError, EmpaiaJobContext
from workflow_execution.contracts import InlineOutput, SharedArtifact, is_json_value


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
        mapping = _mapping(value, "EMPAIA configuration")
        global_values = _json_mapping(mapping.get("global", {}), "global configuration")
        customer_values = _json_mapping(
            mapping.get("customer", {}),
            "customer configuration",
        )
        if set(global_values) & set(customer_values):
            raise _invalid("EMPAIA configuration contains ambiguous values")
        return global_values | customer_values

    async def materialize_input(self, name: str, destination: str) -> None:
        target = Path(destination)
        direct = await self._direct_access(name)
        if direct is not None:
            if not direct.supported:
                raise _invalid("EMPAIA input does not support direct access")
            if not direct.storage_path.is_file():
                raise _unavailable("EMPAIA input is unavailable on shared storage")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(direct.storage_path)
            return
        slide = await self._slide_metadata()
        rectangle = _rectangle(
            await self._request_json(
                "GET",
                f"/v3/{self._job()}/inputs/{quote(name, safe='')}",
            )
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._rectangle_writer.write(target, slide, rectangle)
        except EmpaiaGatewayError:
            raise
        except Exception as error:
            raise _unavailable(
                "EMPAIA rectangle input could not be materialized"
            ) from error

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
        return _required_string(_mapping(value, "EMPAIA pixelmap response"), "id")

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
        mapping = _mapping(_response_json(response), "EMPAIA direct-access response")
        supported = mapping.get("direct_access_supported")
        if not isinstance(supported, bool):
            raise _invalid("EMPAIA direct-access response is invalid")
        return DirectAccess(
            slide_id=_required_string(mapping, "slide_id"),
            storage_path=Path(_required_string(mapping, "storage_path")),
            supported=supported,
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
    mapping = _mapping(value, "EMPAIA slide input")
    extent = _mapping(mapping.get("extent"), "EMPAIA slide extent")
    pixel_size = _mapping(mapping.get("pixel_size_nm"), "EMPAIA slide pixel size")
    raw_levels = mapping.get("levels")
    if not isinstance(raw_levels, list) or not raw_levels:
        raise _invalid("EMPAIA slide levels are invalid")
    levels: list[SlideLevel] = []
    for index, raw_level in enumerate(raw_levels):
        level = _mapping(raw_level, "EMPAIA slide level")
        level_extent = _mapping(level.get("extent"), "EMPAIA slide level extent")
        levels.append(
            SlideLevel(
                index=index,
                width=_positive_int(level_extent, "x"),
                height=_positive_int(level_extent, "y"),
            )
        )
    return SlideMetadata(
        identifier=_required_string(mapping, "id"),
        width=_positive_int(extent, "x"),
        height=_positive_int(extent, "y"),
        pixel_size_nm_x=_positive_number(pixel_size, "x"),
        pixel_size_nm_y=_positive_number(pixel_size, "y"),
        levels=tuple(levels),
    )


def _rectangle(value: object) -> Rectangle:
    mapping = _mapping(value, "EMPAIA rectangle input")
    upper_left = mapping.get("upper_left")
    if (
        not isinstance(upper_left, list)
        or len(upper_left) != 2
        or any(not isinstance(item, int | float) for item in upper_left)
        or any(isinstance(item, bool) for item in upper_left)
    ):
        raise _invalid("EMPAIA rectangle upper_left is invalid")
    return Rectangle(
        x=math.floor(upper_left[0]),
        y=math.floor(upper_left[1]),
        width=math.ceil(_positive_number(mapping, "width")),
        height=math.ceil(_positive_number(mapping, "height")),
    )


def _response_json(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError as error:
        raise _invalid("EMPAIA AppService returned an invalid response") from error


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise _invalid(f"{name} is invalid")
    return value


def _json_mapping(value: object, name: str) -> dict[str, JsonValue]:
    mapping = _mapping(value, name)
    if not is_json_value(mapping):
        raise _invalid(f"{name} is invalid")
    return mapping


def _required_string(value: dict[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise _invalid(f"EMPAIA response has an invalid {name}")
    return item


def _positive_int(value: dict[str, Any], name: str) -> int:
    item = value.get(name)
    if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
        raise _invalid(f"EMPAIA response has an invalid {name}")
    return item


def _positive_number(value: dict[str, Any], name: str) -> float:
    item = value.get(name)
    if (
        not isinstance(item, int | float)
        or isinstance(item, bool)
        or not math.isfinite(item)
        or item <= 0
    ):
        raise _invalid(f"EMPAIA response has an invalid {name}")
    return float(item)


def _invalid(message: str) -> EmpaiaGatewayError:
    return EmpaiaGatewayError(message, retryable=False)


def _unavailable(message: str) -> EmpaiaGatewayError:
    return EmpaiaGatewayError(message, retryable=True)
