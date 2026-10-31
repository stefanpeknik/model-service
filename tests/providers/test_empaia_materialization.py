"""Shared inputs keep the names and companion files slide readers rely on."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from model_service.providers.empaia.http_gateway import DirectAccess
from model_service.providers.empaia.materialization import link_direct_input


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
