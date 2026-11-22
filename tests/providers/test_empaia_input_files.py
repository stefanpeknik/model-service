"""Input files: slides keep the names readers rely on, rectangles become clipped masks."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from model_service.providers.empaia.app_service.gateway import EmpaiaGatewayError
from model_service.providers.empaia.app_service.input_files import (
    link_direct_input,
    write_rectangle_mask,
)
from model_service.providers.empaia.app_service.types import (
    DirectAccess,
    Rectangle,
    SlideLevel,
    SlideMetadata,
)


class LinkDirectInputTests(unittest.TestCase):
    def test_mirax_slide_is_linked_with_its_data_directory(self) -> None:
        with TemporaryDirectory() as directory:
            shared = Path(directory) / "shared"
            (shared / "slide").mkdir(parents=True)
            (shared / "slide.mrxs").write_text("index")
            (shared / "slide" / "Slidedat.ini").write_text("data")
            workspace = Path(directory) / "workspace" / "inputs" / "slide"

            link = link_direct_input(
                DirectAccess(shared / "slide.mrxs", True), workspace
            )

            self.assertEqual(link, workspace / "slide.mrxs")
            self.assertEqual(link.read_text(), "index")
            self.assertEqual((workspace / "slide" / "Slidedat.ini").read_text(), "data")

    def test_single_file_slide_keeps_its_extension(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "scan.svs"
            source.write_text("svs")

            link = link_direct_input(DirectAccess(source, True), Path(directory) / "in")

            self.assertEqual(link.name, "scan.svs")
            self.assertEqual(
                sorted(p.name for p in link.parent.iterdir()), ["scan.svs"]
            )


if __name__ == "__main__":
    unittest.main()


class RectangleMaskTests(unittest.TestCase):
    SLIDE = SlideMetadata(
        "slide-1", 1100, 900, 250.0, 250.0, (SlideLevel(0, 1100, 900),)
    )

    def test_a_rectangle_is_clipped_to_the_slide(self) -> None:
        import pyvips

        with TemporaryDirectory() as directory:
            path = Path(directory) / "mask.tiff"
            write_rectangle_mask(path, self.SLIDE, Rectangle(-500, 800, 1000, 1000))
            mask = pyvips.Image.new_from_file(str(path))

            self.assertEqual((mask.width, mask.height), (1100, 900))
            self.assertEqual(mask.getpoint(0, 800), [255.0])
            self.assertEqual(mask.getpoint(499, 899), [255.0])
            self.assertEqual(mask.getpoint(500, 899), [0.0])
            self.assertEqual(mask.getpoint(0, 799), [0.0])

    def test_a_rectangle_beside_the_slide_is_refused(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mask.tiff"
            with self.assertRaises(EmpaiaGatewayError) as raised:
                write_rectangle_mask(path, self.SLIDE, Rectangle(-500, 0, 400, 100))

            self.assertFalse(raised.exception.retryable)
            self.assertFalse(path.exists())
