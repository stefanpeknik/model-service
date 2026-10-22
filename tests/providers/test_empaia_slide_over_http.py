"""A slide served only through the App API reaches the builder as a readable level."""

from __future__ import annotations

import asyncio
import json
import socket
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
import numpy as np
import pyvips
import uvicorn
from ratiopath.openslide import OpenSlide

from model_service.providers.empaia.http_gateway import HttpEmpaiaGateway, SlideAccess
from testing import empaia_app_service


JOB = "job-over-http"


def write_slide(path: Path) -> dict[str, object]:
    """Write a two-level 0.25 µm/px slide and return its App API metadata."""
    y, x = np.mgrid[0:900, 0:1300]
    pixels = np.stack([x % 256, y % 256, (x // 4) % 256], axis=-1).astype(np.uint8)
    pyvips.Image.new_from_array(pixels).tiffsave(
        str(path),
        tile=True,
        pyramid=True,
        tile_width=256,
        tile_height=256,
        xres=4000,
        yres=4000,
    )
    with OpenSlide(str(path)) as slide:
        levels = [{"extent": {"x": w, "y": h}} for w, h in slide.level_dimensions]
    return {
        "extent": {"x": 1300, "y": 900},
        "pixel_size_nm": {"x": 250.0, "y": 250.0},
        "levels": levels,
    }


class SlideOverHttpTests(unittest.IsolatedAsyncioTestCase):
    async def test_slide_and_region_of_interest_come_from_the_app_api(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.tiff"
            metadata = write_slide(source)
            environment = {
                "MOCK_FILE_SLIDE": str(source),
                "MOCK_FILE_SLIDE_METADATA": json.dumps(metadata),
            }
            with patch.dict("os.environ", environment):
                empaia_app_service._file_slide.cache_clear()
                self.addCleanup(empaia_app_service._file_slide.cache_clear)
                async with serve(empaia_app_service.app) as origin:
                    async with httpx.AsyncClient(base_url=origin) as http:
                        await http.put(f"/validation/jobs/{JOB}/scenario/file")
                    gateway = HttpEmpaiaGateway(
                        origin, JOB, "token", slide_access=SlideAccess.HTTP
                    )
                    try:
                        workspace = Path(directory) / "workspace"
                        workspace.mkdir()
                        slide_path = await gateway.materialize_input(
                            "slide", str(workspace), mpp=0.5
                        )
                        mask_path = await gateway.materialize_input(
                            "region_of_interest", str(workspace), mpp=0.5
                        )
                    finally:
                        await gateway.close()
                    async with httpx.AsyncClient(base_url=origin) as http:
                        observed = (await http.get(f"/validation/jobs/{JOB}")).json()

            with OpenSlide(slide_path) as slide:
                self.assertEqual(slide.level_dimensions, ((650, 450),))
                self.assertAlmostEqual(float(slide.properties["openslide.mpp-x"]), 0.5)
            with OpenSlide(mask_path) as mask:
                self.assertEqual(mask.level_dimensions[0], (1300, 900))
        self.assertEqual(observed["direct_access_reads"], 0)
        # 650x450 in 512-px tiles: 2 columns, 1 row.
        self.assertEqual(observed["region_reads"], 2)


class serve:
    """Run an ASGI app on a loopback port for the duration of a test."""

    def __init__(self, app: object) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self._port = probe.getsockname()[1]
        self._server = uvicorn.Server(
            uvicorn.Config(app, port=self._port, log_level="warning")  # type: ignore[arg-type]
        )

    async def __aenter__(self) -> str:
        self._task = asyncio.create_task(self._server.serve())
        while not self._server.started:
            await asyncio.sleep(0.01)
        return f"http://127.0.0.1:{self._port}"

    async def __aexit__(self, *_exc: object) -> None:
        self._server.should_exit = True
        await self._task


if __name__ == "__main__":
    unittest.main()
