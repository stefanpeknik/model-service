"""Keep raw-inference request and response bytes as files outside the database.

Each execution owns one directory named by its ID: `request`, written before the
row is inserted, and `response-<fencing token>`, written before settlement so a
stale attempt never overwrites the accepted response. Rows store the relative
path and size; a file only becomes visible under its final name once complete.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID, uuid4


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Collection
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class StoredPayload:
    path: str
    size: int


class PayloadMissingError(RuntimeError):
    """A recorded file is absent or does not have its recorded size."""


class PayloadStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    async def write_request(self, execution_id: str, content: bytes) -> StoredPayload:
        return await self._write(f"{execution_id}/request", content)

    async def write_response(
        self, execution_id: str, fencing_token: str, content: bytes
    ) -> StoredPayload:
        return await self._write(f"{execution_id}/response-{fencing_token}", content)

    async def read(self, payload: StoredPayload) -> bytes:
        try:
            content = await asyncio.to_thread(self._resolve(payload.path).read_bytes)
        except FileNotFoundError as error:
            raise PayloadMissingError(payload.path) from error
        if len(content) != payload.size:
            raise PayloadMissingError(payload.path)
        return content

    async def delete(self, paths: Collection[str]) -> None:
        """Remove files or whole execution directories; missing ones are ignored."""
        if paths:
            await asyncio.to_thread(self._delete, [self._resolve(p) for p in paths])

    async def sweep_orphans(
        self,
        existing: Callable[[list[str]], Awaitable[Collection[str]]],
        older_than: float,
    ) -> None:
        """Remove old execution directories that have no execution.

        `existing` reports which candidate IDs still have rows. Such directories
        are left by a crash between writing a request and inserting its row.
        """
        candidates = await asyncio.to_thread(self._old_directories, older_than)
        if not candidates:
            return
        known = set(await existing(candidates))
        await self.delete([name for name in candidates if name not in known])

    async def _write(self, relative: str, content: bytes) -> StoredPayload:
        await asyncio.to_thread(self._write_file, self._resolve(relative), content)
        return StoredPayload(relative, len(content))

    def _resolve(self, relative: str) -> Path:
        path = (self._root / relative).resolve()
        if not path.is_relative_to(self._root.resolve()):
            raise ValueError("payload path escapes the payload root")
        return path

    @staticmethod
    def _write_file(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _delete(paths: list[Path]) -> None:
        for path in paths:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    def _old_directories(self, older_than: float) -> list[str]:
        cutoff = time.time() - older_than
        try:
            entries = list(os.scandir(self._root))
        except FileNotFoundError:
            return []
        # Only execution directories: a volume may hold others, e.g. lost+found.
        return [
            entry.name
            for entry in entries
            if _is_uuid(entry.name)
            and entry.is_dir(follow_symlinks=False)
            and entry.stat(follow_symlinks=False).st_mtime < cutoff
        ]


def _is_uuid(value: str) -> bool:
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False
