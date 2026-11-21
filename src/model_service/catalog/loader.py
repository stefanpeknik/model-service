"""Load a catalog directory for one environment.

Layout, relative to the catalog root:

- `<folder>/<id>.yaml`: one document per file, the file name is its ID; each
  document kind owns one folder (`served-models/`, `analyses/`);
- `environments/<env>/<folder>/<id>.yaml`: a patch holding only what differs
  in that environment, deep-merged over the base file.

Every file is read and every document resolved before anything is reported,
so one load lists every mistake.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import yaml

from model_service.catalog.catalog import Catalog, Entry
from model_service.catalog.context import References, ResolveContext
from model_service.catalog.errors import CatalogError, CatalogInvalid
from model_service.catalog.kinds import DEFAULT_KINDS
from model_service.catalog.merge import deep_merge
from model_service.components.registry import default_registry


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from model_service.catalog.errors import YamlPath
    from model_service.catalog.kinds import DocumentKind
    from model_service.components.registry import Registry


ENVIRONMENTS = "environments"
ID = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAPPING = "A mapping of keys to values"


def environments(root: Path) -> list[str]:
    """The environments that have a patch folder in the catalog.

    Args:
        root: The catalog directory.

    Returns:
        Environment names, sorted.
    """
    folder = root / ENVIRONMENTS
    if not folder.is_dir():
        return []
    return sorted(
        p.name for p in folder.iterdir() if p.is_dir() and not p.name.startswith(".")
    )


def load_catalog(
    root: Path,
    environment: str | None = None,
    *,
    kinds: Sequence[DocumentKind] = DEFAULT_KINDS,
    registry: Registry | None = None,
) -> Catalog:
    """Read, patch and resolve every document of a catalog directory.

    Args:
        root: The catalog directory.
        environment: Whose patches to apply; None for the base files alone.
        kinds: The document kinds, in resolution order.
        registry: The components documents may name; this repo's by default.

    Returns:
        The resolved catalog.

    Raises:
        CatalogInvalid: With every error found, in the standard format.
    """
    loader = _Loader(root, kinds, registry or default_registry())
    if environment is not None and environment not in environments(root):
        loader.file_error(
            root / ENVIRONMENTS / environment,
            "no such environment",
            f"Environments: {', '.join(environments(root)) or 'none'}",
            f"create {loader.show(root / ENVIRONMENTS / environment)}/ or fix the name",
        )
        raise CatalogInvalid(loader.errors)
    catalog = loader.load(environment)
    if loader.errors:
        raise CatalogInvalid(loader.errors)
    return catalog


class _Loader:
    """One load: the files found, the errors so far."""

    def __init__(
        self, root: Path, kinds: Sequence[DocumentKind], registry: Registry
    ) -> None:
        self.root = root
        self.kinds = {kind.folder: kind for kind in kinds}
        self.registry = registry
        self.errors: list[CatalogError] = []

    def show(self, path: Path) -> str:
        """A path as errors show it: relative to the catalog's parent."""
        return str(path.relative_to(self.root.parent))

    def file_error(self, path: Path, what: str, fits: str, hint: str) -> None:
        """Record a mistake about a whole file or folder."""
        self.errors.append(CatalogError(self.show(path), "", what, fits, hint))

    def load(self, environment: str | None) -> Catalog:
        """Resolve every kind's documents in order."""
        bases, patches = self._files(environment)
        references = References(self.kinds)
        for folder, kind in self.kinds.items():
            for path in bases[folder]:
                self._resolve(
                    kind, path, patches.pop((folder, path.stem), None), references
                )
            for (patched_folder, _), path in sorted(patches.items()):
                if patched_folder == folder:
                    known = ", ".join(p.name for p in bases[folder]) or "none"
                    self.file_error(
                        path,
                        "patches a file that does not exist",
                        f"Base files in {folder}/: {known}",
                        "name the patch after the file it changes",
                    )
        return Catalog(environment, references.entries)

    def _files(
        self, environment: str | None
    ) -> tuple[dict[str, list[Path]], dict[tuple[str, str], Path]]:
        """Sort YAML files into base files and this environment's patches.

        Files in no catalog folder are reported.
        """
        bases: dict[str, list[Path]] = {folder: [] for folder in self.kinds}
        patches: dict[tuple[str, str], Path] = {}
        for path in sorted(self.root.rglob("*.yaml")):
            parts = path.relative_to(self.root).parts
            if any(part.startswith(".") for part in parts):
                continue
            if len(parts) == 2 and parts[0] in self.kinds:
                bases[parts[0]].append(path)
            elif (
                len(parts) == 4 and parts[0] == ENVIRONMENTS and parts[2] in self.kinds
            ):
                if parts[1] == environment:
                    patches[(parts[2], path.stem)] = path
            else:
                folders = [f"{f}/" for f in sorted(self.kinds)]
                self.file_error(
                    path,
                    "not a catalog folder",
                    f"Folders: {', '.join([*folders, f'{ENVIRONMENTS}/<env>/'])}",
                    "move the file into one of them",
                )
        return bases, patches

    def _resolve(
        self,
        kind: DocumentKind,
        path: Path,
        patch_path: Path | None,
        references: References,
    ) -> None:
        """Read, patch and resolve one document, recording it or its errors."""
        failed = references.failed[kind.folder]
        if not ID.match(path.stem):
            self.file_error(
                path,
                f"`{path.stem}` is not a valid ID",
                "Lowercase letters, digits and single hyphens, e.g. `my-model`",
                "rename the file; its name is the public ID",
            )
            failed.add(path.stem)
            return
        document = self._read(path)
        patch = self._read(patch_path) if patch_path is not None else {}
        if document is None or patch is None:
            failed.add(path.stem)
            return
        merged, patched = deep_merge(document, patch)

        def file_of(at: YamlPath) -> str:
            hit = any(at[:n] in patched for n in range(1, len(at) + 1))
            return self.show(patch_path if hit and patch_path else path)

        ctx = ResolveContext(self.registry, references, file_of)
        value = kind.resolve(merged, ctx)
        if ctx.errors:
            self.errors += ctx.errors
            failed.add(path.stem)
            return
        references.entries[kind.folder][path.stem] = Entry(
            path.stem, self.show(path), merged, value
        )

    def _read(self, path: Path) -> dict[str, Any] | None:
        """Parse one YAML file into a mapping, or record why it isn't one."""
        try:
            document = yaml.safe_load(path.read_text())
        except yaml.YAMLError as error:
            mark = getattr(error, "problem_mark", None)
            problem = getattr(error, "problem", None) or "unreadable"
            line = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
            near = f" near line {mark.line + 1}" if mark else ""
            self.file_error(
                path,
                f"not valid YAML{line}: {problem}",
                MAPPING,
                f"fix the syntax{near}",
            )
            return None
        if document is None:
            return {}
        if not isinstance(document, dict):
            self.file_error(
                path,
                f"the file is a {_yaml_type(document)}, not a mapping",
                MAPPING,
                "write `key: value` lines at the top level",
            )
            return None
        return document


def _yaml_type(value: object) -> str:
    names: dict[type, str] = {list: "list", str: "string"}
    return names.get(type(value), type(value).__name__)
