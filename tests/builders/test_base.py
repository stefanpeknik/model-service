"""Check that builder artifacts are published atomically."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from model_service.builders.base import _atomic_artifact as atomic_artifact
from model_service.contracts.failures import InvalidInvocation


class AtomicArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_artifact_is_published_atomically(self) -> None:
        destination = self.root / "outputs" / "result.bin"
        with atomic_artifact(destination, inputs=()) as temporary:
            temporary.write_bytes(b"complete")
            self.assertFalse(destination.exists())

        self.assertEqual(destination.read_bytes(), b"complete")
        self.assertFalse(temporary.exists())

    def test_failed_build_leaves_no_partial_artifact(self) -> None:
        destination = self.root / "result.bin"
        with (
            self.assertRaises(OSError),
            atomic_artifact(destination, inputs=()) as temporary,
        ):
            temporary.write_bytes(b"partial")
            raise OSError("storage disconnected")

        self.assertFalse(destination.exists())
        self.assertFalse(temporary.exists())

    def test_output_must_not_replace_an_input(self) -> None:
        source = self.root / "slide.tiff"
        source.write_bytes(b"slide")
        with (
            self.assertRaises(InvalidInvocation),
            atomic_artifact(source, inputs=(source,)),
        ):
            pass

        self.assertEqual(source.read_bytes(), b"slide")


if __name__ == "__main__":
    unittest.main()
