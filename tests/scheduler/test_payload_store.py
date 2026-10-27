"""Payload files stay under their root."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from model_service.scheduler.store.payloads import PayloadStore, StoredPayload


class PayloadStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_paths_cannot_leave_the_root(self) -> None:
        with TemporaryDirectory() as directory:
            store = PayloadStore(Path(directory) / "payloads")
            self.addCleanup(store.close)
            outside = Path(directory) / "outside"
            outside.write_bytes(b"secret")

            for path in ("../outside", "/etc/passwd", ""):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    await store.read(StoredPayload(path, 6))
            with self.assertRaises(ValueError):
                await store.delete(["../outside"])
            self.assertTrue(outside.exists())


if __name__ == "__main__":
    unittest.main()
