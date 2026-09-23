from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from fastapi import FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel


class JobView(BaseModel):
    configuration_reads: int = 0
    rectangle_reads: int = 0
    tile_count: int = 0
    finalized: bool = False
    failure: str | None = None


@dataclass(slots=True)
class JobRecord:
    configuration_reads: int = 0
    rectangle_reads: int = 0
    tile_count: int = 0
    output_created: bool = False
    finalized: bool = False
    failure: str | None = None


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


@app.get("/v3/{job_id}/configuration")
def configuration(
    job_id: str, authorization: str | None = Header(default=None)
) -> dict:
    _authorize(authorization)
    with _lock:
        _job(job_id).configuration_reads += 1
    return {"global": {}, "customer": {}}


@app.get("/v3/rationai/{job_id}/inputs/{name}/direct-access")
def direct_access(
    job_id: str,
    name: str,
    authorization: str | None = Header(default=None),
) -> dict:
    _authorize(authorization)
    if name not in {"slide", "region_of_interest"}:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if name == "region_of_interest" and job_id.startswith("rectangle-"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    with _lock:
        _job(job_id)
    filename = "slide.dat" if name == "slide" else "roi.dat"
    return {
        "direct_access_supported": True,
        "slide_id": f"{job_id}-slide",
        "storage_path": f"/mnt/validation/input/{filename}",
    }


@app.get("/v3/{job_id}/inputs/slide")
def slide(job_id: str, authorization: str | None = Header(default=None)) -> dict:
    _authorize(authorization)
    return {
        "id": f"{job_id}-slide",
        "extent": {"x": 8, "y": 8},
        "pixel_size_nm": {"x": 250, "y": 250},
        "levels": [{"extent": {"x": 8, "y": 8}}],
    }


@app.get("/v3/{job_id}/inputs/region_of_interest")
def rectangle(
    job_id: str,
    authorization: str | None = Header(default=None),
) -> dict:
    _authorize(authorization)
    with _lock:
        _job(job_id).rectangle_reads += 1
    return {"upper_left": [1, 1], "width": 6, "height": 6}


@app.post("/v3/{job_id}/outputs/{name}")
def create_output(
    job_id: str,
    name: str,
    body: dict,
    authorization: str | None = Header(default=None),
) -> dict:
    _authorize(authorization)
    if name != "heatmap" or body.get("type") != "continuous_pixelmap":
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)
    with _lock:
        _job(job_id).output_created = True
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
    if (
        pixelmap_id != f"{job_id}-heatmap"
        or level != 0
        or tile_x != 0
        or tile_y != 0
        or not await request.body()
    ):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)
    with _lock:
        if not _job(job_id).output_created:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT)
        _job(job_id).tile_count += 1


@app.put("/v3/{job_id}/finalize", status_code=status.HTTP_204_NO_CONTENT)
def finalize(job_id: str, authorization: str | None = Header(default=None)) -> None:
    _authorize(authorization)
    with _lock:
        record = _job(job_id)
        if not record.output_created or record.tile_count == 0:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT)
        record.finalized = True


@app.put("/v3/{job_id}/failure", status_code=status.HTTP_204_NO_CONTENT)
def failure(
    job_id: str,
    body: dict,
    authorization: str | None = Header(default=None),
) -> None:
    _authorize(authorization)
    with _lock:
        _job(job_id).failure = str(body.get("user_message", ""))


@app.get("/validation/jobs/{job_id}", response_model=JobView)
def inspect_job(job_id: str) -> JobView:
    with _lock:
        record = _jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return JobView(
            configuration_reads=record.configuration_reads,
            rectangle_reads=record.rectangle_reads,
            tile_count=record.tile_count,
            finalized=record.finalized,
            failure=record.failure,
        )
