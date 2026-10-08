"""Check TIFF-to-pixelmap publication through the mock AppService."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import httpx

from integrations.empaia.http_gateway import HttpEmpaiaGateway
from validation.empaia_app_service import app
from workflow_execution.contracts import SharedArtifact


try:
    import pyvips
except (ImportError, OSError):
    HAS_VIPS = False
else:
    HAS_VIPS = bool(pyvips.Image)


@unittest.skipUnless(HAS_VIPS, "libvips is unavailable")
class ValidationEmpaiaPublicationTests(unittest.IsolatedAsyncioTestCase):
    """Exercise TIFF-to-pixelmap publication through the mock AppService."""

    async def test_tiff_pixelmap_publication(self) -> None:
        """Verify TIFF pixelmap publication.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heatmap.tiff"
            pyvips.Image.black(8, 8).new_from_image(128).tiffsave(str(path))
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://appservice",
            ) as client:
                gateway = HttpEmpaiaGateway(
                    "http://appservice",
                    "validation-publication",
                    "synthetic-token",
                    client=client,
                )
                await gateway.publish_output(
                    "heatmap", SharedArtifact(path=str(path), media_type="image/tiff")
                )
                await gateway.finalize()


if __name__ == "__main__":
    unittest.main()
