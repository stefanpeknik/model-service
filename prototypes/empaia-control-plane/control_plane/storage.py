from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True, slots=True)
class StorageMapping:
    """Deployment-owned translation from EMPAIA storage paths to worker locators."""

    storage_id: str
    source_prefix: str
    mount_root: Path

    def __post_init__(self) -> None:
        if not self.storage_id or any(
            character.isspace() for character in self.storage_id
        ):
            raise ValueError("storage_id must be a non-empty identifier")
        _validate_relative_path(self.source_prefix)
        if not self.mount_root.is_absolute():
            raise ValueError("storage mount root must be absolute")
        if not self.mount_root.is_dir():
            raise ValueError("storage mount root must be an existing directory")


@dataclass(frozen=True, slots=True)
class AuthorizedLocator:
    storage_id: str
    relative_locator: str


class InvalidStorageInputError(ValueError):
    pass


class StorageUnavailableError(RuntimeError):
    pass


class StorageLocatorResolver:
    def __init__(self, mappings: tuple[StorageMapping, ...]) -> None:
        if not mappings:
            raise ValueError("at least one storage mapping is required")
        if len({mapping.storage_id for mapping in mappings}) != len(mappings):
            raise ValueError("storage IDs must be unique")
        prefixes = [PurePosixPath(mapping.source_prefix) for mapping in mappings]
        for index, prefix in enumerate(prefixes):
            for other in prefixes[index + 1 :]:
                if _contains(prefix, other) or _contains(other, prefix):
                    raise ValueError("storage source prefixes must not overlap")
        self._mappings = mappings

    def resolve(self, storage_path: str) -> AuthorizedLocator:
        try:
            _validate_relative_path(storage_path)
        except ValueError as error:
            raise InvalidStorageInputError(str(error)) from error
        path = PurePosixPath(storage_path)
        matches: list[tuple[StorageMapping, PurePosixPath]] = []
        for mapping in self._mappings:
            prefix = PurePosixPath(mapping.source_prefix)
            try:
                relative = path.relative_to(prefix)
            except ValueError:
                continue
            matches.append((mapping, relative))
        if len(matches) != 1:
            raise InvalidStorageInputError(
                "storage path must match exactly one configured mapping"
            )

        mapping, relative = matches[0]
        if str(relative) in {"", "."}:
            raise InvalidStorageInputError(
                "storage path must identify an object below its prefix"
            )
        try:
            root = mapping.mount_root.resolve(strict=True)
        except OSError as error:
            raise StorageUnavailableError(
                "configured storage is unavailable"
            ) from error
        try:
            candidate = root
            candidate_stat = None
            for part in relative.parts:
                candidate /= part
                candidate_stat = candidate.lstat()
                if stat.S_ISLNK(candidate_stat.st_mode):
                    raise InvalidStorageInputError(
                        "storage locator must not contain symbolic links"
                    )
            candidate = candidate.resolve(strict=True)
            candidate.relative_to(root)
        except InvalidStorageInputError:
            raise
        except FileNotFoundError as error:
            raise InvalidStorageInputError(
                "storage locator does not identify an existing object"
            ) from error
        except ValueError as error:
            raise InvalidStorageInputError(
                "storage locator escapes its configured mount"
            ) from error
        except OSError as error:
            raise StorageUnavailableError(
                "configured storage is unavailable"
            ) from error
        if candidate_stat is None or not stat.S_ISREG(candidate_stat.st_mode):
            raise InvalidStorageInputError(
                "storage locator does not identify a regular file"
            )
        return AuthorizedLocator(mapping.storage_id, str(candidate.relative_to(root)))


def _validate_relative_path(value: str) -> None:
    raw_parts = value.split("/")
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise ValueError("storage path must be a traversal-free POSIX relative path")


def _contains(parent: PurePosixPath, child: PurePosixPath) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True
