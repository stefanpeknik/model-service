"""Shared inputs keep the names and companion files slide readers rely on."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from model_service.providers.empaia.app_service.input_files import (
    _clip,
    link_direct_input,
)
from model_service.providers.empaia.app_service.types import DirectAccess


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


class RectangleTests(unittest.TestCase):
    def test_a_rectangle_is_clipped_to_the_slide(self) -> None:
        self.assertEqual(_clip(10, 20, 100), (10, 20))
        self.assertEqual(_clip(-100, 200, 1000), (0, 100))
        self.assertEqual(_clip(90, 20, 100), (90, 10))
        # Nothing on the slide: not positive, so the writer refuses it.
        self.assertLessEqual(_clip(-500, 100, 1000)[1], 0)


if __name__ == "__main__":
    unittest.main()
