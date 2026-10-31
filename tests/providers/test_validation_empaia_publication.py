"""Check score-grid publication through the mock AppService."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import httpx
import numpy as np
from aiohttp import web
from aiohttp.test_utils import TestServer

from model_service.builders.contracts import SharedArtifact
from model_service.builders.score_grid import MEDIA_TYPE, ScoreGrid, write_score_grid
from model_service.providers.empaia.http_gateway import HttpEmpaiaGateway
from testing.empaia_app_service import app


class ValidationEmpaiaPublicationTests(unittest.IsolatedAsyncioTestCase):
    async def test_score_grid_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heatmap.npz"
            # The mock's slide: 8x8 pixels in one level, so 4-pixel cells.
            write_score_grid(
                path,
                ScoreGrid(
                    np.full((1, 2, 2), 0.5, dtype=np.float32),
                    np.ones((2, 2), dtype=bool),
                    0,
                    (8, 8),
                    4.0,
                    (0.25, 0.25),
                ),
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://appservice",
            ) as client:

                async def forward(request: web.Request) -> web.Response:
                    """Bridge the SDK's loopback HTTP calls to the existing ASGI mock."""
                    response = await client.request(
                        request.method,
                        request.path_qs,
                        headers=request.headers,
                        content=await request.read(),
                    )
                    return web.Response(
                        status=response.status_code,
                        headers=response.headers,
                        body=response.content,
                    )

                server_app = web.Application()
                server_app.router.add_route("*", "/{path:.*}", forward)
                async with TestServer(server_app) as server:
                    gateway = HttpEmpaiaGateway(
                        str(server.make_url("/")),
                        "validation-publication",
                        "synthetic-token",
                    )
                    try:
                        await gateway.publish_output(
                            "heatmap",
                            SharedArtifact(path=str(path), media_type=MEDIA_TYPE),
                        )
                        await gateway.finalize()
                    finally:
                        await gateway.close()


if __name__ == "__main__":
    unittest.main()
