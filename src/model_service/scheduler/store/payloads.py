"""Keep raw-inference request and response bytes as files outside the database.

An execution's files sit in one of 256 shard directories named by the first two
characters of its ID: `<id>.request`, written before the row is inserted, and
`<id>.response-<fencing token>`, written before settlement so a stale attempt
never overwrites the accepted response. Rows store the relative path and size.
A file is written under a temporary name and renamed into place.

Creating, renaming or deleting a file locks its directory, and on a network file
system each such operation takes tens of milliseconds; shards keep concurrent
tiles from queueing on one directory.
"""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import PurePosixPath
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
    def __init__(self, root: Path, *, io_threads: int = 32) -> None:
        """Use a dedicated thread pool for file I/O under the payload root.

        Args:
            root: Directory holding the shard directories.
            io_threads: Threads for file operations; the default executor would queue slow ones.
        """
        self._root = root
        # The default executor (CPUs + 4 threads) would queue slow file operations.
        self._io = ThreadPoolExecutor(io_threads, thread_name_prefix="payload-io")

    def close(self) -> None:
        """Shut down the I/O thread pool without waiting for running operations."""
        self._io.shutdown(wait=False)

    async def write_request(self, execution_id: str, content: bytes) -> StoredPayload:
        """Write an execution's request file.

        Args:
            execution_id: ID of the execution.
            content: Request body.

        Returns:
            The file's relative path and size.

        Raises:
            OSError: The file cannot be written.
        """
        return await self._write(_name(execution_id, "request"), content)

    async def write_response(
        self, execution_id: str, fencing_token: str, content: bytes
    ) -> StoredPayload:
        """Write an attempt's response file, named by its fencing token so a stale attempt never overwrites the accepted one.

        Args:
            execution_id: ID of the execution.
            fencing_token: Token of the attempt that produced the response.
            content: Response body.

        Returns:
            The file's relative path and size.

        Raises:
            OSError: The file cannot be written.
        """
        return await self._write(
            _name(execution_id, f"response-{fencing_token}"), content
        )

    async def read(self, payload: StoredPayload) -> bytes:
        """Read a stored file and check it has its recorded size.

        Args:
            payload: Recorded path and size.

        Returns:
            The file's contents.

        Raises:
            PayloadMissingError: The file is absent or its size differs from the record.
            OSError: The file cannot be read.
        """
        try:
            content = await self._run(self._resolve(payload.path).read_bytes)
        except FileNotFoundError as error:
            raise PayloadMissingError(payload.path) from error
        if len(content) != payload.size:
            raise PayloadMissingError(payload.path)
        return content

    async def delete(self, paths: Collection[str]) -> None:
        """Remove files; missing ones are ignored.

        Args:
            paths: Relative paths of the files.

        Raises:
            ValueError: A path escapes the payload root.
            OSError: A file cannot be removed.
        """
        resolved = [self._resolve(path) for path in paths]
        await asyncio.gather(*(self._run(_unlink, path) for path in resolved))

    async def sweep_orphans(
        self,
        existing: Callable[[list[str]], Awaitable[Collection[str]]],
        older_than: float,
    ) -> None:
        """Remove files older than `older_than` seconds that belong to no execution.

        Such files are left by a crash between writing a file and recording it.

        Args:
            existing: Callback reporting which of the given IDs still have rows; called with up to 1000 IDs at a time.
            older_than: Minimum file age in seconds.

        Raises:
            OSError: A shard cannot be listed or a file cannot be removed.
        """
        shards = [f"{value:02x}" for value in range(256)]
        listings = await asyncio.gather(*(self._run(self._list, s) for s in shards))
        files: dict[str, list[str]] = {}
        for listing in listings:
            for path in listing:
                files.setdefault(PurePosixPath(path).name[:36], []).append(path)
        ids = list(files)
        known: set[str] = set()
        for start in range(0, len(ids), 1000):
            known.update(await existing(ids[start : start + 1000]))
        unowned = [path for value in ids if value not in known for path in files[value]]
        cutoff = time.time() - older_than
        modified = await asyncio.gather(
            *(self._run(self._modified, path) for path in unowned)
        )
        await self.delete(
            [
                path
                for path, at in zip(unowned, modified, strict=True)
                if at is not None and at < cutoff
            ]
        )

    async def _write(self, relative: str, content: bytes) -> StoredPayload:
        """Write a file under a relative path.

        Args:
            relative: Path relative to the payload root.
            content: File contents.

        Returns:
            The file's relative path and size.
        """
        await self._run(_write_file, self._resolve(relative), content)
        return StoredPayload(relative, len(content))

    async def _run[*Ts, T](self, function: Callable[[*Ts], T], *args: *Ts) -> T:
        """Run a blocking function on the I/O thread pool.

        Args:
            function: Function to run.
            *args: Arguments passed to function.

        Returns:
            The function's return value.
        """
        return await asyncio.get_running_loop().run_in_executor(
            self._io, function, *args
        )

    def _resolve(self, relative: str) -> Path:
        """Turn a stored relative path into a path under the root.

        Args:
            relative: Path relative to the payload root.

        Returns:
            The path under the root.

        Raises:
            ValueError: The path is empty, absolute, or contains `..`.
        """
        # Lexical only: resolving would stat every component on the event loop.
        path = PurePosixPath(relative)
        if not path.parts or path.is_absolute() or ".." in path.parts:
            raise ValueError("payload path escapes the payload root")
        return self._root.joinpath(*path.parts)

    def _list(self, shard: str) -> list[str]:
        """List the execution files in one shard directory.

        Args:
            shard: Two-character shard name.

        Returns:
            Relative paths of files whose names start with an execution ID; empty if the shard does not exist.
        """
        try:
            names = os.listdir(self._root / shard)
        except FileNotFoundError:
            return []
        # Only execution files: a volume may hold others, e.g. lost+found.
        return [f"{shard}/{name}" for name in names if _is_uuid(name[:36])]

    def _modified(self, relative: str) -> float | None:
        """Return a file's modification time.

        Args:
            relative: Path relative to the payload root.

        Returns:
            The modification time in seconds since the epoch, or None if the file is gone.
        """
        try:
            return self._resolve(relative).stat().st_mtime
        except FileNotFoundError:
            return None


def _write_file(path: Path, content: bytes) -> None:
    """Write a file atomically through a temporary name, fsynced before the rename.

    Creates the shard directory on first use; the temporary file is removed on failure.

    Args:
        path: Final path of the file.
        content: File contents.
    """
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        file = temporary.open("wb")
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
        file = temporary.open("wb")
    try:
        with file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _unlink(path: Path) -> None:
    """Remove a file if it exists.

    Args:
        path: Path of the file.
    """
    path.unlink(missing_ok=True)


def _name(execution_id: str, kind: str) -> str:
    """Build an execution file's relative path in its shard.

    Args:
        execution_id: ID of the execution; its first two characters name the shard.
        kind: Suffix such as `request` or `response-<token>`.

    Returns:
        The path `<shard>/<id>.<kind>`.
    """
    return f"{execution_id[:2]}/{execution_id}.{kind}"


def _is_uuid(value: str) -> bool:
    """Return whether a value is a UUID in canonical form.

    Args:
        value: Candidate execution ID.

    Returns:
        True if the value round-trips through UUID unchanged.
    """
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False
