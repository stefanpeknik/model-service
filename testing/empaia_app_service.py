"""Provide an in-memory AppService mock with observable calls and failure scenarios.

Job ID prefixes select deterministic behavior for deployment tests. This mock checks
bearer-header presence only and does not establish real EMPAIA authentication or
interoperability.

Environment:
    MOCK_INPUT_ROOT: directory holding the synthetic fixtures (default
        `/mnt/validation/input`).
    MOCK_FILE_SLIDE, MOCK_FILE_ROI, MOCK_FILE_SLIDE_METADATA: for `file-` jobs,
        the paths of a real slide and region mask and the slide's metadata as
        JSON (`extent`, `pixel_size_nm`, `levels`), as the App API returns it.
        `file-` jobs also serve the slide through the region route, for
        model-service's `EMPAIA_SLIDE_ACCESS=http`.
    MOCK_FILE_RECTANGLE: for `file-` jobs, the region of interest returned as a
        rectangle input (`upper_left`, `width`, `height`); defaults to the whole
        slide.
"""

from __future__ import annotations

import asyncio
import functools
import io
import json
import os
import struct
import uuid
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock

from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict


class ValidationScenario(StrEnum):
    """Name the job ID prefixes that trigger delays, publication failures, restarts, or native fixture behavior."""

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
    BUILDER_FAILURE = "builder-failure"
    FILE = "file"


def _stable_id(value: str) -> str:
    """A UUID derived from `value`, shaped as the version 4 the SDK's models require."""
    return str(uuid.UUID(bytes=uuid.uuid5(uuid.NAMESPACE_URL, value).bytes, version=4))


def _slide_id(job_id: str) -> str:
    """The real service identifies slides and pixelmaps by UUID."""
    return _stable_id(f"{job_id}-slide")


def _pixelmap_id(job_id: str) -> str:
    return _stable_id(f"{job_id}-heatmap")


class PixelmapRequest(BaseModel):
    """Accept the output type needed by the mock heatmap creation endpoint.

    Attributes:
        type: Mock output representation, required to be continuous_pixelmap for
            heatmaps.
    """

    model_config = ConfigDict(extra="allow")

    type: str


class FailureRequest(BaseModel):
    """Accept the safe user message sent by provider terminal reporting.

    Attributes:
        user_message: Safe provider failure/cancellation report text.
    """

    user_message: str


class JobView(BaseModel):
    """Expose call counters, sampled tile values, and terminal state for test assertions.

    Attributes:
        configuration_reads: Number of provider configuration GETs observed.
        direct_access_reads: Number of direct-access input probes observed.
        rectangle_reads: Number of region rectangle GETs observed.
        region_reads: Number of slide region image GETs observed.
        slide_metadata_reads: Number of slide geometry GETs observed.
        output_create_attempts: Number of attempted external pixelmap registrations.
        tile_attempts: Number of attempted tile uploads, including failures.
        tile_count: Number of successfully accepted tile uploads.
        tile_sample: Sampled native heatmap value from the uploaded float32 payload.
        tile_background_sample: Sampled excluded-region value used by rectangle
            validation.
        finalize_attempts: Number of external finalization requests observed.
        finalize_ack_blocked: Whether the mock currently withholds finalization
            acknowledgment.
        finalized: Whether the mock job has accepted finalization.
        failure: Latest provider failure/cancellation message.
        failure_reports: Number of terminal failure reports observed.
    """

    configuration_reads: int = 0
    direct_access_reads: int = 0
    rectangle_reads: int = 0
    region_reads: int = 0
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
    """Keep mutable per-job mock state, including coordination flags used by restart scenarios.

    Attributes:
        configuration_reads: Number of provider configuration GETs observed.
        direct_access_reads: Number of direct-access input probes observed.
        rectangle_reads: Number of region rectangle GETs observed.
        region_reads: Number of slide region image GETs observed.
        slide_metadata_reads: Number of slide geometry GETs observed.
        output_create_attempts: Number of attempted external pixelmap registrations.
        tile_attempts: Number of attempted tile uploads, including failures.
        tile_count: Number of successfully accepted tile uploads.
        tile_sample: Sampled native heatmap value from the uploaded float32 payload.
        tile_background_sample: Sampled excluded-region value used by rectangle
            validation.
        finalize_attempts: Number of external finalization requests observed.
        output_created: Whether external pixelmap registration occurred.
        metadata_released: Whether the restart test released blocked slide metadata.
        finalize_ack_blocked: Whether the mock currently withholds finalization
            acknowledgment.
        finalize_ack_released: Whether the test released the withheld acknowledgment.
        finalized: Whether the mock job has accepted finalization.
        failure: Latest provider failure/cancellation message.
        failure_reports: Number of terminal failure reports observed.
    """

    configuration_reads: int = 0
    direct_access_reads: int = 0
    rectangle_reads: int = 0
    region_reads: int = 0
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
# Scenarios declared for job IDs without a prefix, e.g. UUIDs required by JES.
_scenarios: dict[str, ValidationScenario] = {}
_lock = Lock()


def _authorize(authorization: str | None) -> None:
    """Require a nonempty Bearer header without validating the credential signature.

    Args:
        authorization: Incoming bearer header checked by the validation mock.

    Returns:
        None: No return value.

    Raises:
        HTTPException: The request fails the route's safe validation, availability, or
            execution contract.
    """
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    if not authorization.removeprefix("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


def _job(job_id: str) -> JobRecord:
    """Get or create a mock record; callers hold the shared lock when updating it.

    Args:
        job_id: External EMPAIA job identifier.

    Returns:
        JobRecord: Existing or newly created mutable mock record; callers coordinate
            updates with the shared lock.
    """
    return _jobs.setdefault(job_id, JobRecord())


def _is_scenario(job_id: str, scenario: ValidationScenario) -> bool:
    """Match a job ID against a scenario prefix followed by a dash.

    Args:
        job_id: External EMPAIA job identifier.
        scenario: Validation job prefix selecting deterministic mock behavior.

    Returns:
        bool: True when the job ID begins with the selected scenario prefix.
    """
    return job_id.startswith(f"{scenario.value}-") or _scenarios.get(job_id) is scenario


def _is_native(job_id: str) -> bool:
    """Identify scenarios using real synthetic TIFF inputs instead of dummy byte fixtures.

    Args:
        job_id: External EMPAIA job identifier.

    Returns:
        bool: True when the job uses the native processor or native rectangle fixture.
    """
    return _is_scenario(job_id, ValidationScenario.NATIVE_PROCESSOR) or _is_scenario(
        job_id, ValidationScenario.NATIVE_RECTANGLE
    )


@app.get("/v3/{job_id}/configuration")
def configuration(
    job_id: str, authorization: str | None = Header(default=None)
) -> dict[str, object]:
    """Count configuration reads and return scenario-specific values or authentication/service failures.

    Args:
        job_id: External EMPAIA job identifier.
        authorization: Incoming bearer header checked by the validation mock.

    Returns:
        dict[str, object]: Decoded, serialized, or recorded fields described above.

    Raises:
        HTTPException: The request fails the route's safe validation, availability, or
            execution contract.
    """
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
    return {"global": {}, "customer": customer}


@app.get("/v3/{job_id}/mode")
def mode(
    job_id: str, authorization: str | None = Header(default=None)
) -> dict[str, object]:
    """Report every mock job as a standalone one."""
    _authorize(authorization)
    return {"mode": "STANDALONE"}


@app.get("/v3/rationai/{job_id}/slides/{name}/direct-access")
async def direct_access(
    job_id: str,
    name: str,
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    """Return fixture paths, with scenario-controlled delays, missing inputs, or a 404 to exercise rectangle fallback.

    Args:
        job_id: External EMPAIA job identifier.
        name: Input/output name or configuration field selected by this operation.
        authorization: Incoming bearer header checked by the validation mock.

    Returns:
        dict[str, object]: Decoded, serialized, or recorded fields described above.

    Raises:
        HTTPException: The request fails the route's safe validation, availability, or
            execution contract.
    """
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
    if _is_scenario(job_id, ValidationScenario.FILE):
        storage_path = os.environ[
            "MOCK_FILE_SLIDE" if name == "slide" else "MOCK_FILE_ROI"
        ]
    else:
        filename = "slide.dat" if name == "slide" else "roi.dat"
        if _is_native(job_id):
            filename = "slide.tiff" if name == "slide" else "roi.tiff"
        if name == "slide" and _is_scenario(job_id, ValidationScenario.BUILDER_FAILURE):
            filename = "missing-slide.dat"
        root = os.environ.get("MOCK_INPUT_ROOT", "/mnt/validation/input")
        storage_path = f"{root}/{filename}"
    return {
        "direct_access_supported": True,
        "slide_id": _slide_id(job_id),
        # As CDS: absolute under its data directory, and relative to it.
        "physical_path": storage_path,
        "storage_path": storage_path.lstrip("/"),
    }


@app.get("/v3/{job_id}/inputs/slide")
async def slide(
    job_id: str, authorization: str | None = Header(default=None)
) -> dict[str, object]:
    """Return dummy or native slide geometry, optionally blocking publication until a restart test releases metadata.

    Args:
        job_id: External EMPAIA job identifier.
        authorization: Incoming bearer header checked by the validation mock.

    Returns:
        dict[str, object]: Decoded, serialized, or recorded fields described above.
    """
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
    if _is_scenario(job_id, ValidationScenario.FILE):
        return _complete_slide(
            _slide_id(job_id), json.loads(os.environ["MOCK_FILE_SLIDE_METADATA"])
        )
    native = _is_native(job_id)
    extent = 256 if native else 8
    pixel_size_nm = 500 if native else 250
    return _complete_slide(
        _slide_id(job_id),
        {
            "extent": {"x": extent, "y": extent},
            "pixel_size_nm": {"x": pixel_size_nm, "y": pixel_size_nm},
            "levels": [{"extent": {"x": extent, "y": extent}}],
        },
    )


def _complete_slide(slide_id: str, metadata: dict[str, object]) -> dict[str, object]:
    """Add what the real service sends and the SDK's models require."""
    extent = {**metadata["extent"], "z": 1}  # type: ignore[dict-item]
    levels = [
        {
            "extent": {**level["extent"], "z": 1},
            "downsample_factor": extent["x"] / level["extent"]["x"],
        }
        for level in metadata["levels"]  # type: ignore[attr-defined]
    ]
    return {
        "id": slide_id,
        "extent": extent,
        "num_levels": len(levels),
        "pixel_size_nm": metadata["pixel_size_nm"],
        "tile_extent": {"x": 256, "y": 256, "z": 1},
        "levels": levels,
        "format": "mock",
        "tissue": None,
        "stain": None,
    }


@app.get("/v3/{job_id}/inputs/region_of_interest")
def rectangle(
    job_id: str,
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    """Count region reads and return the fixed slide-space rectangle used by mask tests.

    Args:
        job_id: External EMPAIA job identifier.
        authorization: Incoming bearer header checked by the validation mock.

    Returns:
        dict[str, object]: Decoded, serialized, or recorded fields described above.
    """
    _authorize(authorization)
    with _lock:
        _job(job_id).rectangle_reads += 1
    if _is_scenario(job_id, ValidationScenario.FILE):
        if "MOCK_FILE_RECTANGLE" in os.environ:
            return json.loads(os.environ["MOCK_FILE_RECTANGLE"])
        extent = json.loads(os.environ["MOCK_FILE_SLIDE_METADATA"])["extent"]
        return {"upper_left": [0, 0], "width": extent["x"], "height": extent["y"]}
    return {"upper_left": [1, 1], "width": 6, "height": 6}


@app.get(
    "/v3/{job_id}/regions/{wsi_id}/level/{level}/start/{start_x}/{start_y}"
    "/size/{size_x}/{size_y}"
)
async def region(
    job_id: str,
    wsi_id: str,
    level: int,
    start_x: int,
    start_y: int,
    size_x: int,
    size_y: int,
    authorization: str | None = Header(default=None),
) -> Response:
    """Serve a PNG region of the `file-` job's slide; level coordinates as in the App API."""
    _authorize(authorization)
    with _lock:
        _job(job_id).region_reads += 1
    if not _is_scenario(job_id, ValidationScenario.FILE) or wsi_id != _slide_id(job_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    data = await asyncio.to_thread(
        _read_region, level, start_x, start_y, size_x, size_y
    )
    return Response(content=data, media_type="image/png")


@functools.cache
def _file_slide() -> object:
    import openslide

    return openslide.OpenSlide(os.environ["MOCK_FILE_SLIDE"])


def _read_region(level: int, x: int, y: int, width: int, height: int) -> bytes:
    slide = _file_slide()
    if not 0 <= level < slide.level_count:  # type: ignore[attr-defined]
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    downsample = slide.level_downsamples[level]  # type: ignore[attr-defined]
    image = slide.read_region(  # type: ignore[attr-defined]
        (round(x * downsample), round(y * downsample)), level, (width, height)
    )
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


@app.post("/v3/{job_id}/outputs/{name}")
def create_output(
    job_id: str,
    name: str,
    body: PixelmapRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    """Require a continuous heatmap pixelmap, count its creation, and return a stable mock output ID.

    Args:
        job_id: External EMPAIA job identifier.
        name: Input/output name or configuration field selected by this operation.
        body: Mock pixelmap output type used to validate heatmap creation.
        authorization: Incoming bearer header checked by the validation mock.

    Returns:
        dict[str, object]: Decoded, serialized, or recorded fields described above.

    Raises:
        HTTPException: The request fails the route's safe validation, availability, or
            execution contract.
    """
    _authorize(authorization)
    if name != "heatmap" or body.type != "continuous_pixelmap":
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)
    with _lock:
        record = _job(job_id)
        record.output_create_attempts += 1
        record.output_created = True
    return {
        **body.model_dump(),
        "id": _pixelmap_id(job_id),
        "created_at": 1,
        "updated_at": 1,
    }


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
    """Validate tile coordinates and payload, record native sample values, and optionally fail publication.

    Args:
        job_id: External EMPAIA job identifier.
        pixelmap_id: External output pixelmap identifier returned by AppService.
        level: Selected pyramid level index.
        tile_x: Horizontal pixelmap tile position.
        tile_y: Vertical pixelmap tile position.
        request: Request to validate, admit, dispatch, forward, or process, as specified
            by its declared type.
        authorization: Incoming bearer header checked by the validation mock.

    Returns:
        None: No return value.

    Raises:
        HTTPException: The request fails the route's safe validation, availability, or
            execution contract.
    """
    _authorize(authorization)
    data = await request.body()
    # A real slide's heatmap spans many tiles on the level matching the model.
    single_tile = not _is_scenario(job_id, ValidationScenario.FILE)
    if (
        pixelmap_id != _pixelmap_id(job_id)
        or (single_tile and (level != 0 or tile_x != 0 or tile_y != 0))
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
    """Require published output tiles and record success, optionally failing or withholding the first acknowledgment.

    Args:
        job_id: External EMPAIA job identifier.
        authorization: Incoming bearer header checked by the validation mock.

    Returns:
        None: No return value.

    Raises:
        HTTPException: The request fails the route's safe validation, availability, or
            execution contract.
    """
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
    """Record a provider failure or cancellation report and increment its call count.

    Args:
        job_id: External EMPAIA job identifier.
        body: Safe external failure message to record in mock job state.
        authorization: Incoming bearer header checked by the validation mock.

    Returns:
        None: No return value.
    """
    _authorize(authorization)
    with _lock:
        record = _job(job_id)
        record.failure = body.user_message
        record.failure_reports += 1


@app.get("/validation/jobs/{job_id}", response_model=JobView)
def inspect_job(job_id: str) -> JobView:
    """Return a locked snapshot of observable job state or 404 for an unknown job.

    Args:
        job_id: External EMPAIA job identifier.

    Returns:
        JobView: Observable mock call counters, tile samples, and terminal job state.

    Raises:
        HTTPException: The request fails the route's safe validation, availability, or
            execution contract.
    """
    with _lock:
        record = _jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return JobView(
            configuration_reads=record.configuration_reads,
            direct_access_reads=record.direct_access_reads,
            rectangle_reads=record.rectangle_reads,
            region_reads=record.region_reads,
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


@app.put("/validation/jobs/{job_id}/scenario/{scenario}", status_code=204)
def declare_scenario(job_id: str, scenario: ValidationScenario) -> None:
    """Select a scenario for a job whose ID cannot carry a prefix."""
    _scenarios[job_id] = scenario


@app.post("/validation/jobs/{job_id}/release-metadata", status_code=204)
def release_metadata(job_id: str) -> None:
    """Unblock the metadata read in a provider-finish restart scenario.

    Args:
        job_id: External EMPAIA job identifier.

    Returns:
        None: No return value.

    Raises:
        HTTPException: The request fails the route's safe validation, availability, or
            execution contract.
    """
    if not _is_scenario(job_id, ValidationScenario.FINISH_RESTART):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    with _lock:
        _job(job_id).metadata_released = True


@app.post("/validation/jobs/{job_id}/release-finalize", status_code=204)
def release_finalize(job_id: str) -> None:
    """Unblock the withheld acknowledgment in a finalization outcome-unknown scenario.

    Args:
        job_id: External EMPAIA job identifier.

    Returns:
        None: No return value.

    Raises:
        HTTPException: The request fails the route's safe validation, availability, or
            execution contract.
    """
    if not _is_scenario(job_id, ValidationScenario.FINALIZATION_ACK_LOST):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    with _lock:
        record = _job(job_id)
        record.finalize_ack_released = True
        record.finalize_ack_blocked = False
