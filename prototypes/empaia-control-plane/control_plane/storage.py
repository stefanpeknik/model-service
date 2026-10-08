"""Authorize provider storage locators through deployment-controlled mount mappings.

Historical SQLite control-plane prototype. The current production scheduler lives in
execution_scheduler/ and uses PostgreSQL; these lifecycle rules are not its
specification.
"""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True, slots=True)
class StorageMapping:
    """Deployment-owned translation from EMPAIA storage paths to worker locators.

    Attributes:
        storage_id: Configured worker storage mapping identifier.
        source_prefix: Traversal-free provider path prefix permitted by a prototype
            storage mapping.
        mount_root: Existing absolute worker directory for the storage translation.
    """

    storage_id: str
    source_prefix: str
    mount_root: Path

    def __post_init__(self) -> None:
        """Require a storage ID, traversal-free source prefix, and existing absolute mount directory.

        Raises:
            ValueError: A value fails the validation described above.
        """
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
    """Carry a configured storage ID and a checked relative worker locator.

    Attributes:
        storage_id: Configured worker storage mapping identifier.
        relative_locator: Traversal-free file locator beneath the authorized worker
            mount.
    """

    storage_id: str
    relative_locator: str


class InvalidStorageInputError(ValueError):
    """Signal a provider path outside the permitted mapping or an unsupported file representation."""


class StorageUnavailableError(RuntimeError):
    """Signal temporary access failure to a configured storage mount."""


class StorageLocatorResolver:
    """Translate provider paths into contained regular-file locators without allowing symlink components."""

    def __init__(self, mappings: tuple[StorageMapping, ...]) -> None:
        """Require nonempty mappings with unique storage IDs and disjoint provider path prefixes.

        Args:
            mappings: Configured app contracts or provider-to-worker storage
                translations.

        Raises:
            ValueError: A value fails the validation described above.
        """
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
        """Select exactly one mapping and verify every path component, containment, and regular-file type before returning a locator.

        Args:
            storage_path: Provider storage locator to authorize or expose in fixture
                metadata.

        Returns:
            AuthorizedLocator: Authorized storage ID and contained relative file
                locator.

        Raises:
            InvalidStorageInputError: The locator is invalid, ambiguous, missing, a
                symlink, outside its mount, or not a regular file.
            StorageUnavailableError: The configured mount or object cannot be accessed.
        """
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
    """Reject empty, absolute, or traversal-containing provider POSIX paths.

    Args:
        value: Value being validated, serialized, converted, or checked by this helper.

    Returns:
        None: No return value.

    Raises:
        ValueError: A value fails the validation described above.
    """
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
    """Check whether a POSIX path is equal to or nested beneath another path.

    Args:
        parent: Containing POSIX path for the containment check.
        child: Path checked for containment beneath the parent.

    Returns:
        bool: True when child is equal to or nested beneath parent.
    """
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True
