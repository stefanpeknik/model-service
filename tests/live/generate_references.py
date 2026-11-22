"""Record golden references from a running deployment.

Raw routes get tiles read locally from copies of the reference slides; runs go
through the API with the slides on the deployment's shared storage, and their
outputs stay there (copy them next to the raw references, then `--manifest`):

    uv run python -m tests.live.generate_references raw \
        --serve-url http://localhost:18000 --slides .cache/golden/slides \
        --out .cache/golden/refs --model prostate-classifier-1
    uv run python -m tests.live.generate_references run \
        --api-url http://localhost:18080 --shared-root /mnt/shared/golden \
        --analysis prostate-heatmap
    uv run python -m tests.live.generate_references manifest \
        --out .cache/golden/refs --image-tag empaia-36 --source-revision 0352fa6
"""

from __future__ import annotations

import argparse
import ast
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tests.live.golden import MANIFEST, RAW_CASES, RUN_CASES, RawCase, sha256


if TYPE_CHECKING:
    import httpx
    from numpy.typing import NDArray

# Slides under the shared root's `slides/`; the local copy has the same names.
SLIDES = (
    "67df0411-395d-5d73-9c2f-3e0ffef45491.mrxs",
    "97b654fd-d0b4-58e2-84d6-7b599f1ed331.mrxs",
)
TILES_PER_SLIDE = 16
UNPOOLED_TILES = 4


def tissue_positions(
    slide_path: Path, case: RawCase, count: int
) -> tuple[int, list[tuple[int, int]]]:
    """Pick tile positions on tissue, deterministically.

    A thumbnail cell darker than the glass counts as tissue; this only picks
    sample tiles for references, it is not the service's tissue detection.

    Args:
        slide_path: The slide.
        case: The model whose tile size and resolution are used.
        count: How many positions to pick.

    Returns:
        The level read and level-0 (x, y) positions of tile corners.
    """
    import openslide

    with openslide.OpenSlide(str(slide_path)) as slide:
        mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        level = slide.get_best_level_for_downsample(case.mpp / mpp)
        downsample = slide.level_downsamples[level]
        span = int(case.tile_size * downsample)
        width, height = slide.dimensions
        thumbnail = np.asarray(
            slide.get_thumbnail((width // span, height // span)).convert("L"),
            dtype=np.float32,
        )
    rows, columns = np.nonzero(thumbnail < 200)
    rng = np.random.default_rng(0)
    picks = rng.choice(len(rows), size=min(count, len(rows)), replace=False)
    scale_x = width / thumbnail.shape[1]
    scale_y = height / thumbnail.shape[0]
    positions = [
        (int(columns[i] * scale_x), int(rows[i] * scale_y)) for i in sorted(picks)
    ]
    return level, positions


def read_tile(
    slide_path: Path, level: int, x: int, y: int, size: int
) -> NDArray[np.uint8]:
    """Read one RGB tile.

    Args:
        slide_path: The slide.
        level: Level to read from.
        x: Level-0 x of the corner.
        y: Level-0 y of the corner.
        size: Tile edge in level pixels.

    Returns:
        The tile, (size, size, 3) uint8.
    """
    import openslide

    with openslide.OpenSlide(str(slide_path)) as slide:
        region = slide.read_region((x, y), level, (size, size)).convert("RGB")
    return np.asarray(region, dtype=np.uint8)


def call_raw(
    client: httpx.Client, case: RawCase, tile: NDArray[np.uint8], *, pooled: bool = True
) -> NDArray[np.generic]:
    """Send a tile to a served model's raw route and decode the reply.

    Args:
        client: Client with the Serve base URL.
        case: The served model.
        tile: The RGB tile.
        pooled: For embeddings, whether the route pools the tokens.

    Returns:
        The decoded reply.
    """
    import lz4.frame

    headers = {} if pooled else {"x-pool-tokens": "false"}
    response = client.post(
        f"/{case.model}/",
        content=lz4.frame.compress(tile.tobytes()),
        headers=headers,
    )
    response.raise_for_status()
    if case.kind == "score":
        return np.asarray(response.json(), dtype=np.float64)
    data = lz4.frame.decompress(response.content)
    if case.kind == "score_map":
        return np.frombuffer(data, dtype=case.dtype).copy()
    shape = ast.literal_eval(response.headers["x-output-shape"])
    return np.frombuffer(data, dtype=np.float32).reshape(shape).copy()


def record_raw(args: argparse.Namespace) -> None:
    """Record one served model's raw-route references.

    Args:
        args: The parsed `raw` command line.
    """
    import httpx

    case = RAW_CASES[args.model]
    out = args.out / "raw" / case.model
    out.mkdir(parents=True, exist_ok=True)
    tiles, sources = [], []
    for name in SLIDES:
        slide_path = args.slides / name
        level, positions = tissue_positions(slide_path, case, TILES_PER_SLIDE)
        for x, y in positions:
            tiles.append(read_tile(slide_path, level, x, y, case.tile_size))
            sources.append({"slide": name, "level": level, "x": x, "y": y})
    replies: dict[str, NDArray[np.generic]] = {}
    with httpx.Client(base_url=args.serve_url, timeout=args.timeout) as client:
        for i, tile in enumerate(tiles):
            replies[f"tile_{i:03d}"] = call_raw(client, case, tile)
        if case.kind == "embedding" and case.model == "virchow2":
            for i, tile in enumerate(tiles[:UNPOOLED_TILES]):
                replies[f"unpooled_{i:03d}"] = call_raw(
                    client, case, tile, pooled=False
                )
    np.save(out / "tiles.npy", np.stack(tiles))
    np.savez(out / "replies.npz", **replies)
    (out / "case.json").write_text(
        json.dumps({"model": case.model, "kind": case.kind, "tiles": sources}, indent=2)
        + "\n"
    )
    print(f"{case.model}: {len(tiles)} tiles, {len(replies)} replies -> {out}")


def record_run(args: argparse.Namespace) -> None:
    """Run one analysis's builder and model synchronously on each slide.

    Args:
        args: The parsed `run` command line.
    """
    import httpx

    case = RUN_CASES[args.analysis]
    deadline = datetime.now(UTC) + timedelta(seconds=args.timeout)
    with httpx.Client(base_url=args.api_url, timeout=args.timeout + 60) as client:
        for name in SLIDES:
            output = f"{args.prefix}/{case.analysis}/{Path(name).stem}.{case.extension}"
            response = client.post(
                f"/v1/builders/{case.builder}/models/{case.model}/run",
                json={
                    "inputs": {
                        "slide": {"kind": "shared_path", "path": f"slides/{name}"}
                    },
                    "input_access": {
                        "kind": "shared_storage",
                        "state": {"root": args.shared_root},
                    },
                    "output_path": output,
                    "deadline_at": deadline.isoformat(),
                },
            )
            print(case.analysis, name, response.status_code, response.text[:500])
            response.raise_for_status()


def write_manifest(args: argparse.Namespace) -> None:
    """Hash every reference file and write the committed manifest.

    Args:
        args: The parsed `manifest` command line.
    """
    files = {
        str(path.relative_to(args.out)): sha256(path)
        for path in sorted(args.out.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "recorded": datetime.now(UTC).date().isoformat(),
        "image_tag": args.image_tag,
        "source_revision": args.source_revision,
        "shared_root": args.shared_root,
        "slides": list(SLIDES),
        "models": json.loads(args.models.read_text()) if args.models else {},
        "files": files,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(files)} files -> {MANIFEST}")


def main() -> None:
    """Parse the command line and run the chosen recording step."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    raw = commands.add_parser("raw", help="raw-route replies of one served model")
    raw.add_argument("--serve-url", required=True)
    raw.add_argument("--slides", type=Path, required=True)
    raw.add_argument("--out", type=Path, required=True)
    raw.add_argument("--model", choices=sorted(RAW_CASES), required=True)
    raw.add_argument("--timeout", type=float, default=1800)
    raw.set_defaults(handler=record_raw)
    run = commands.add_parser("run", help="synchronous runs of one analysis")
    run.add_argument("--api-url", required=True)
    run.add_argument("--shared-root", default="/mnt/shared/golden")
    run.add_argument(
        "--prefix", default="runs", help="output folder under the shared root"
    )
    run.add_argument("--analysis", choices=sorted(RUN_CASES), required=True)
    run.add_argument("--timeout", type=float, default=7200)
    run.set_defaults(handler=record_run)
    manifest = commands.add_parser("manifest", help="hash the local reference set")
    manifest.add_argument("--out", type=Path, required=True)
    manifest.add_argument("--image-tag", required=True)
    manifest.add_argument("--source-revision", required=True)
    manifest.add_argument("--shared-root", default="/mnt/shared/golden")
    manifest.add_argument(
        "--models", type=Path, help="JSON of model versions (weights URIs, revisions)"
    )
    manifest.set_defaults(handler=write_manifest)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
