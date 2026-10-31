"""Check shared-storage root allowlisting, path containment and attempt isolation."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from model_service.builders.contracts import (
    ExecutionContext,
    InvalidInvocation,
    SharedPathInput,
)
from model_service.providers.shared_storage import SharedStorageAccessAdapter
from tests.support.builder_fixtures import invocation


def shared(root: str | Path, **inputs: str):  # type: ignore[no-untyped-def]
    return invocation(
        "shared_storage",
        {"root": str(root)},
        inputs={name: SharedPathInput(path) for name, path in inputs.items()},
    )


class SharedStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.allowed = Path(self._directory.name) / "allowed"
        self.outside = Path(self._directory.name) / "outside"
        (self.allowed / "project").mkdir(parents=True)
        self.outside.mkdir()
        (self.allowed / "project" / "slide.tiff").write_bytes(b"slide")
        (self.outside / "secret.tiff").write_bytes(b"secret")
        self.adapter = SharedStorageAccessAdapter([self.allowed])

    def test_allowed_root_and_its_subdirectories_are_accepted(self) -> None:
        for root, path in (
            (self.allowed, "project/slide.tiff"),
            (self.allowed / "project", "slide.tiff"),
        ):
            with self.subTest(root=root):
                access = self.adapter.create(
                    shared(root, slide=path), ExecutionContext()
                )
                self.assertEqual(
                    access.input_path("slide"),
                    (self.allowed / "project" / "slide.tiff").resolve(),
                )

    def test_roots_outside_the_allowlist_are_rejected(self) -> None:
        cases = {
            "outside": self.outside,
            "parent": self.allowed.parent,
            "dot-dot": self.allowed / ".." / "outside",
        }
        for name, root in cases.items():
            with (
                self.subTest(name),
                self.assertRaisesRegex(InvalidInvocation, "not allowed"),
            ):
                self.adapter.create(shared(root), ExecutionContext())

    def test_symlinked_root_into_a_forbidden_tree_is_rejected(self) -> None:
        link = self.allowed / "link"
        link.symlink_to(self.outside, target_is_directory=True)
        with self.assertRaisesRegex(InvalidInvocation, "not allowed"):
            self.adapter.create(shared(link), ExecutionContext())

    def test_empty_allowlist_rejects_every_root(self) -> None:
        with self.assertRaisesRegex(InvalidInvocation, "not allowed"):
            SharedStorageAccessAdapter([]).create(
                shared(self.allowed), ExecutionContext()
            )

    def test_missing_or_file_roots_are_rejected(self) -> None:
        for root in (self.allowed / "missing", self.allowed / "project" / "slide.tiff"):
            with self.subTest(root=root), self.assertRaises(InvalidInvocation):
                self.adapter.create(shared(root), ExecutionContext())

    def test_state_must_contain_only_a_non_empty_string_root(self) -> None:
        for state in (
            {},
            {"root": None},
            {"root": True},
            {"root": 1},
            {"root": ""},
            {"root": str(self.allowed), "credential": "secret"},
            {"root": str(self.allowed), "processing_attempt_id": "caller-attempt"},
        ):
            with self.subTest(state=state), self.assertRaises(InvalidInvocation):
                self.adapter.create(
                    invocation("shared_storage", state), ExecutionContext()
                )

    def test_input_paths_cannot_escape_the_root(self) -> None:
        (self.allowed / "project" / "escape.tiff").symlink_to(
            self.outside / "secret.tiff"
        )
        (self.allowed / "project" / "folder").mkdir()
        for path in (
            "../outside/secret.tiff",
            "/etc/passwd",
            "a/./b",
            "a//b",
            "back\\slash",
            "escape.tiff",
            "folder",
            "missing.tiff",
        ):
            access = self.adapter.create(
                shared(self.allowed / "project", slide=path), ExecutionContext()
            )
            with self.subTest(path=path), self.assertRaises(InvalidInvocation):
                access.input_path("slide")

    def test_provider_inputs_are_not_shared_paths(self) -> None:
        access = self.adapter.create(shared(self.allowed), ExecutionContext())
        with self.assertRaisesRegex(InvalidInvocation, "not a shared path"):
            access.input_path("slide")

    def test_attempt_outputs_are_isolated_under_a_hash_of_the_attempt(self) -> None:
        def access(attempt: str | None):  # type: ignore[no-untyped-def]
            return self.adapter.create(
                shared(self.allowed),
                ExecutionContext(processing_attempt_id=attempt),
            )

        old = access("old").output_path("outputs/heatmap.tiff")
        new = access("new").output_path("outputs/heatmap.tiff")
        digest = hashlib.sha256(b"new").hexdigest()
        self.assertEqual(
            new,
            self.allowed.resolve() / ".attempts" / digest / "outputs" / "heatmap.tiff",
        )
        self.assertNotEqual(old, new)
        new.parent.mkdir(parents=True)
        new.write_bytes(b"accepted")
        old.parent.mkdir(parents=True)
        old.write_bytes(b"late old attempt")
        self.assertEqual(new.read_bytes(), b"accepted")
        self.assertTrue(access("new").is_visible_artifact(new))
        self.assertFalse(
            access("new").is_visible_artifact(self.outside / "secret.tiff")
        )
        with self.assertRaises(InvalidInvocation):
            access("new").output_path("../escape.tiff")
        self.assertEqual(
            access(None).output_path("outputs/heatmap.tiff"),
            self.allowed.resolve() / "outputs" / "heatmap.tiff",
        )


if __name__ == "__main__":
    unittest.main()
