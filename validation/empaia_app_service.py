from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock

from fastapi import FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel


class ValidationScenario(StrEnum):
    AUTH_REJECTED = "auth-rejected"
    CANCELLATION = "cancellation"
    DIRECT_ACCESS = "direct-access"
    FINALIZATION_FAILURE = "finalization-failure"
    FINALIZATION_ACK_LOST = "finalization-ack-lost"
    FINISH_RESTART = "finish-restart"
    INVALID_CONFIGURATION = "invalid-configuration"
    NATIVE_PROCESSOR = "native-processor"
    NATIVE_RECTANGLE = "native-rectangle"
    PUBLICATION_FAILURE = "publication-failure"
    RAY_RESTART = "ray-restart"
    RECTANGLE = "rectangle"
    SCHEDULER_RESTART = "scheduler-restart"
    SERVICE_UNAVAILABLE = "service-unavailable"
    WORKFLOW_FAILURE = "workflow-failure"


class PixelmapRequest(BaseModel):
    type: str


class FailureRequest(BaseModel):
    user_message: str


class JobView(BaseModel):
    configuration_reads: int = 0
    direct_access_reads: int = 0
    rectangle_reads: int = 0
    slide_metadata_reads: int = 0
    output_create_attempts: int = 0
    tile_attempts: int = 0
    tile_count: int = 0
    tile_sample: float | None = None
    tile_background_sample: float | None = None
    finalize_attempts: int = 0
    finalize_ack_blocked: bool = False
    finalized: bool = False
    failure: str | None = None
    failure_reports: int = 0


@dataclass(slots=True)
class JobRecord:
    configuration_reads: int = 0
    direct_access_reads: int = 0
    rectangle_reads: int = 0
    slide_metadata_reads: int = 0
    output_create_attempts: int = 0
    tile_attempts: int = 0
    tile_count: int = 0
    tile_sample: float | None = None
    tile_background_sample: float | None = None
    finalize_attempts: int = 0
    output_created: bool = False
    metadata_released: bool = False
    finalize_ack_blocked: bool = False
    finalize_ack_released: bool = False
    finalized: bool = False
    failure: str | None = None
    failure_reports: int = 0


app = FastAPI()
_jobs: dict[str, JobRecord] = {}
_lock = Lock()


def _authorize(authorization: str | None) -> None:
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    if not authorization.removeprefix("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


def _job(job_id: str) -> JobRecord:
    return _jobs.setdefault(job_id, JobRecord())


def _is_scenario(job_id: str, scenario: ValidationScenario) -> bool:
    return job_id.startswith(f"{scenario.value}-")


def _is_native(job_id: str) -> bool:
    return _is_scenario(job_id, ValidationScenario.NATIVE_PROCESSOR) or _is_scenario(
        job_id, ValidationScenario.NATIVE_RECTANGLE
    )


@app.get("/v3/{job_id}/configuration")
def configuration(
    job_id: str, authorization: str | None = Header(default=None)
) -> dict[str, object]:
    _authorize(authorization)
    with _lock:
        _job(job_id).configuration_reads += 1
    if _is_scenario(job_id, ValidationScenario.AUTH_REJECTED):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    if _is_scenario(job_id, ValidationScenario.SERVICE_UNAVAILABLE):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    customer: dict[str, object] = {}
    if _is_scenario(job_id, ValidationScenario.INVALID_CONFIGURATION):
        customer["unexpected"] = True
    if _is_scenario(job_id, ValidationScenario.CANCELLATION):
        customer["output_path"] = "cancellation/heatmap.tiff"
    return {"global": {}, "customer": customer}


@app.get("/v3/rationai/{job_id}/inputs/{name}/direct-access")
async def direct_access(
    job_id: str,
    name: str,
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    _authorize(authorization)
    if name not in {"slide", "region_of_interest"}:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    with _lock:
        record = _job(job_id)
        record.direct_access_reads += 1
        direct_access_reads = record.direct_access_reads
    if name == "slide" and _is_scenario(job_id, ValidationScenario.CANCELLATION):
        await asyncio.sleep(300)
    if (
        name == "slide"
        and direct_access_reads == 1
        and (
            _is_scenario(job_id, ValidationScenario.RAY_RESTART)
            or _is_scenario(job_id, ValidationScenario.SCHEDULER_RESTART)
        )
    ):
        await asyncio.sleep(45)
    if name == "region_of_interest" and (
        _is_scenario(job_id, ValidationScenario.RECTANGLE)
        or _is_scenario(job_id, ValidationScenario.NATIVE_RECTANGLE)
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    filename = "slide.dat" if name == "slide" else "roi.dat"
    if _is_native(job_id):
        filename = "slide.tiff" if name == "slide" else "roi.tiff"
    if name == "slide" and _is_scenario(job_id, ValidationScenario.WORKFLOW_FAILURE):
        filename = "missing-slide.dat"
    return {
        "direct_access_supported": True,
        "slide_id": f"{job_id}-slide",
        "storage_path": f"/mnt/validation/input/{filename}",
    }


@app.get("/v3/{job_id}/inputs/slide")
async def slide(
    job_id: str, authorization: str | None = Header(default=None)
) -> dict[str, object]:
    _authorize(authorization)
    with _lock:
        record = _job(job_id)
        record.slide_metadata_reads += 1
    if _is_scenario(job_id, ValidationScenario.FINISH_RESTART):
        while True:
            with _lock:
                if _job(job_id).metadata_released:
                    break
            await asyncio.sleep(0.1)
    native = _is_native(job_id)
    extent = 256 if native else 8
    pixel_size_nm = 500 if native else 250
    return {
        "id": f"{job_id}-slide",
        "extent": {"x": extent, "y": extent},
        "pixel_size_nm": {"x": pixel_size_nm, "y": pixel_size_nm},
        "levels": [{"extent": {"x": extent, "y": extent}}],
    }


@app.get("/v3/{job_id}/inputs/region_of_interest")
def rectangle(
    job_id: str,
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    _authorize(authorization)
    with _lock:
        _job(job_id).rectangle_reads += 1
    return {"upper_left": [1, 1], "width": 6, "height": 6}


@app.post("/v3/{job_id}/outputs/{name}")
def create_output(
    job_id: str,
    name: str,
    body: PixelmapRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    _authorize(authorization)
    if name != "heatmap" or body.type != "continuous_pixelmap":
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)
    with _lock:
        record = _job(job_id)
        record.output_create_attempts += 1
        record.output_created = True
    return {"id": f"{job_id}-heatmap"}


@app.put(
    "/v3/{job_id}/pixelmaps/{pixelmap_id}/level/{level}/position/{tile_x}/{tile_y}/data",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def put_tile(
    job_id: str,
    pixelmap_id: str,
    level: int,
    tile_x: int,
    tile_y: int,
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    _authorize(authorization)
    data = await request.body()
    if (
        pixelmap_id != f"{job_id}-heatmap"
        or level != 0
        or tile_x != 0
        or tile_y != 0
        or not data
    ):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)
    tile_sample = None
    tile_background_sample = None
    if _is_native(job_id):
        if len(data) != 512 * 512 * 4:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)
        sample_at = (
            4 if _is_scenario(job_id, ValidationScenario.NATIVE_RECTANGLE) else 128
        )
        tile_sample = struct.unpack_from("=f", data, 4 * (sample_at * 512 + sample_at))[
            0
        ]
        if _is_scenario(job_id, ValidationScenario.NATIVE_RECTANGLE):
            tile_background_sample = struct.unpack_from(
                "=f", data, 4 * (200 * 512 + 200)
            )[0]
    with _lock:
        record = _job(job_id)
        record.tile_attempts += 1
        if not record.output_created:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT)
        if _is_scenario(job_id, ValidationScenario.PUBLICATION_FAILURE):
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        record.tile_count += 1
        record.tile_sample = tile_sample
        record.tile_background_sample = tile_background_sample


@app.put("/v3/{job_id}/finalize", status_code=status.HTTP_204_NO_CONTENT)
async def finalize(
    job_id: str, authorization: str | None = Header(default=None)
) -> None:
    _authorize(authorization)
    with _lock:
        record = _job(job_id)
        record.finalize_attempts += 1
        if _is_scenario(job_id, ValidationScenario.FINALIZATION_FAILURE):
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        if not record.output_created or record.tile_count == 0:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT)
        record.finalized = True
        blocked = (
            _is_scenario(job_id, ValidationScenario.FINALIZATION_ACK_LOST)
            and record.finalize_attempts == 1
        )
        if blocked:
            record.finalize_ack_blocked = True
    if blocked:
        while True:
            with _lock:
                if _job(job_id).finalize_ack_released:
                    break
            await asyncio.sleep(0.1)


@app.put("/v3/{job_id}/failure", status_code=status.HTTP_204_NO_CONTENT)
def failure(
    job_id: str,
    body: FailureRequest,
    authorization: str | None = Header(default=None),
) -> None:
    _authorize(authorization)
    with _lock:
        record = _job(job_id)
        record.failure = body.user_message
        record.failure_reports += 1


@app.get("/validation/jobs/{job_id}", response_model=JobView)
def inspect_job(job_id: str) -> JobView:
    with _lock:
        record = _jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return JobView(
            configuration_reads=record.configuration_reads,
            direct_access_reads=record.direct_access_reads,
            rectangle_reads=record.rectangle_reads,
            slide_metadata_reads=record.slide_metadata_reads,
            output_create_attempts=record.output_create_attempts,
            tile_attempts=record.tile_attempts,
            tile_count=record.tile_count,
            tile_sample=record.tile_sample,
            tile_background_sample=record.tile_background_sample,
            finalize_attempts=record.finalize_attempts,
            finalize_ack_blocked=record.finalize_ack_blocked,
            finalized=record.finalized,
            failure=record.failure,
            failure_reports=record.failure_reports,
        )


@app.post("/validation/jobs/{job_id}/release-metadata", status_code=204)
def release_metadata(job_id: str) -> None:
    if not _is_scenario(job_id, ValidationScenario.FINISH_RESTART):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    with _lock:
        _job(job_id).metadata_released = True


@app.post("/validation/jobs/{job_id}/release-finalize", status_code=204)
def release_finalize(job_id: str) -> None:
    if not _is_scenario(job_id, ValidationScenario.FINALIZATION_ACK_LOST):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    with _lock:
        record = _job(job_id)
        record.finalize_ack_released = True
        record.finalize_ack_blocked = False
