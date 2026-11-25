"""Load a catalog directory, optionally patched for one environment.

The catalog root holds `models/<id>.yaml` and `pipelines/<id>.yaml`:
one document per file, the file name is its ID. An environment is a folder of
its own (in this repo `deploy/helm/environments/<env>/`) whose
`<folder>/<id>.yaml` files are patches holding only what differs on that
cluster, deep-merged over the base file; its other files (Helm values, worker
groups, manifests) are not the loader's.

Models resolve first, so pipelines can call them. Every file is read
and every document resolved before anything is reported, so one load lists
every mistake.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import yaml

from model_service.catalog import models as model_files
from model_service.catalog import pipelines
from model_service.catalog.catalog import Catalog
from model_service.catalog.context import ResolveContext
from model_service.catalog.errors import CatalogError, CatalogInvalid
from model_service.catalog.merge import deep_merge
from model_service.registry import default_registry


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from model_service.catalog.errors import YamlPath
    from model_service.catalog.models import Model
    from model_service.registry import Registry


ID = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
MAPPING = "A mapping of keys to values"
FOLDERS = (model_files.FOLDER, pipelines.FOLDER)
# What else an environment folder holds, which the loader leaves to Helm.
ENVIRONMENT_FILES = ("values.yaml", "secrets.yaml", "workers/")


def environments(folder: Path) -> list[Path]:
    """The environment folders in a folder of environments.

    Args:
        folder: E.g. `deploy/helm/environments`.

    Returns:
        Each environment's folder, sorted by name; none if `folder` is missing.
    """
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_dir() and not p.name.startswith(".")
    )


def load_catalog(
    root: Path,
    environment: Path | None = None,
    *,
    registry: Registry | None = None,
) -> Catalog:
    """Read, patch and resolve every document of a catalog directory.

    Args:
        root: The catalog directory.
        environment: The folder of the environment whose patches to apply;
            None for the base files alone.
        registry: The components documents may name; this repo's by default.

    Returns:
        The resolved catalog.

    Raises:
        CatalogInvalid: With every error found, in the standard format.
    """
    loader = _Loader(root, registry or default_registry())
    if environment is not None and not environment.is_dir():
        known = ", ".join(p.name for p in environments(environment.parent)) or "none"
        loader.file_error(
            environment,
            "no such environment",
            f"Environments: {known}",
            f"create {loader.show(environment)}/ or fix the name",
        )
        raise CatalogInvalid(loader.errors)
    catalog = loader.load(environment)
    if loader.errors:
        raise CatalogInvalid(loader.errors)
    return catalog


class _Loader:
    """One load: the files found, the errors so far."""

    def __init__(self, root: Path, registry: Registry) -> None:
        """Start a load of one catalog root.

        Args:
            root: The catalog folder.
            registry: The components files may name.
        """
        self.root = root
        self.registry = registry
        self.errors: list[CatalogError] = []

    def show(self, path: Path) -> str:
        """A path as errors show it: relative to the catalog's parent.

        Args:
            path: A file or folder in the catalog.

        Returns:
            The shown path; absolute when it lies outside the catalog's parent.
        """
        return (
            str(path.relative_to(self.root.parent))
            if path.is_relative_to(self.root.parent)
            else str(path)
        )

    def file_error(self, path: Path, what: str, fits: str, hint: str) -> None:
        """Record a mistake about a whole file or folder.

        Args:
            path: The file or folder.
            what: What is wrong.
            fits: What would fit.
            hint: How to fix it.
        """
        self.errors.append(CatalogError(self.show(path), "", what, fits, hint))

    def load(self, environment: Path | None) -> Catalog:
        """Resolve the models, then the pipelines.

        Args:
            environment: The environment folder whose patches to apply, or None.

        Returns:
            The catalog; `self.errors` holds what failed.
        """
        bases, patches = self._files(environment)
        folder = model_files.FOLDER
        models = self._folder(
            folder, model_files.build_model, bases, patches, {}, set()
        )
        failed = {path.stem for path in bases[folder]} - models.keys()
        found = self._folder(
            pipelines.FOLDER, pipelines.build_pipeline, bases, patches, models, failed
        )
        return Catalog(models, found)

    def _folder[T](
        self,
        folder: str,
        build: Callable[[dict[str, Any], ResolveContext], T | None],
        bases: dict[str, list[Path]],
        patches: dict[tuple[str, str], Path],
        models: Mapping[str, Model],
        failed: set[str],
    ) -> dict[str, T]:
        """Resolve every document of one folder and report stray patches.

        Args:
            folder: `models` or `pipelines`.
            build: Resolves one merged document through a context.
            bases: Base files by folder.
            patches: This environment's patches by (folder, file stem).
            models: The models a pipeline may call.
            failed: IDs of model files that failed to resolve.

        Returns:
            The resolved documents by ID.
        """
        resolved: dict[str, T] = {}
        for path in bases[folder]:
            patch = patches.get((folder, path.stem))
            value = self._resolve(build, path, patch, models, failed)
            if value is not None:
                resolved[path.stem] = value
        for (patched_folder, stem), path in sorted(patches.items()):
            if patched_folder == folder and stem not in {p.stem for p in bases[folder]}:
                known = ", ".join(p.name for p in bases[folder]) or "none"
                self.file_error(
                    path,
                    "patches a file that does not exist",
                    f"Base files in {folder}/: {known}",
                    "name the patch after the file it changes",
                )
        return resolved

    def _files(
        self, environment: Path | None
    ) -> tuple[dict[str, list[Path]], dict[tuple[str, str], Path]]:
        """Sort YAML files into base files and this environment's patches.

        Files in no catalog folder are reported.

        Args:
            environment: The environment folder whose patches to pick, or None.

        Returns:
            Base files by folder, and patches by (folder, file stem).
        """
        bases: dict[str, list[Path]] = {folder: [] for folder in FOLDERS}
        patches: dict[tuple[str, str], Path] = {}
        for path in sorted(self.root.rglob("*.yaml")):
            parts = path.relative_to(self.root).parts
            if any(part.startswith(".") for part in parts):
                continue
            if len(parts) == 2 and parts[0] in FOLDERS:
                bases[parts[0]].append(path)
            else:
                folders = [f"{f}/" for f in sorted(FOLDERS)]
                self.file_error(
                    path,
                    "not a catalog folder",
                    f"Folders: {', '.join(folders)}",
                    "move the file into one of them",
                )
        if environment is not None:
            for path in sorted(environment.rglob("*.y*ml")):
                parts = path.relative_to(environment).parts
                if any(part.startswith(".") for part in parts):
                    continue
                if len(parts) == 2 and parts[0] in FOLDERS and path.suffix == ".yaml":
                    patches[(parts[0], path.stem)] = path
                elif parts[0] not in ("values.yaml", "secrets.yaml", "workers"):
                    folders = [f"{f}/" for f in sorted(FOLDERS)]
                    self.file_error(
                        path,
                        "not a patch folder",
                        f"Patch folders: {', '.join(folders)}, beside "
                        f"{', '.join(ENVIRONMENT_FILES)}",
                        "move the file into one of them",
                    )
        return bases, patches

    def _resolve[T](
        self,
        build: Callable[[dict[str, Any], ResolveContext], T | None],
        path: Path,
        patch_path: Path | None,
        models: Mapping[str, Model],
        failed_models: set[str],
    ) -> T | None:
        """Read, patch and resolve one document, recording its errors.

        Args:
            build: Resolves the merged document through a context.
            path: The base file.
            patch_path: The environment's patch of it, if any.
            models: The models a pipeline may call.
            failed_models: IDs of models that failed to resolve.

        Returns:
            The resolved document, or None after an error.
        """
        if not ID.fullmatch(path.stem):
            self.file_error(
                path,
                f"`{path.stem}` is not a valid ID",
                "Lowercase letters, digits and single hyphens, e.g. `my-model`",
                "rename the file; its name is the public ID",
            )
            return None
        document = self._read(path)
        patch = self._read(patch_path) if patch_path is not None else {}
        if document is None or patch is None:
            return None
        merged, patched = deep_merge(document, patch)

        def file_of(at: YamlPath) -> str:
            """The file an error at a YAML path belongs to: the patch if it set that path.

            Args:
                at: The YAML path.

            Returns:
                The shown path of the base file or the patch.
            """
            hit = any(at[:n] in patched for n in range(1, len(at) + 1))
            return self.show(patch_path if hit and patch_path else path)

        ctx = ResolveContext(self.registry, file_of, path.stem, models, failed_models)
        value = build(merged, ctx)
        if ctx.errors:
            self.errors += ctx.errors
            return None
        return value

    def _read(self, path: Path) -> dict[str, Any] | None:
        """Parse one YAML file into a mapping, or record why it isn't one.

        Args:
            path: The file.

        Returns:
            The mapping, empty for an empty file, or None after an error.
        """
        if path.is_dir():
            self.file_error(
                path, "a folder, not a file", MAPPING, "rename or remove the folder"
            )
            return None
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as error:
            bad_line = data[: error.start].count(b"\n") + 1
            self.file_error(
                path,
                f"not UTF-8 text: byte 0x{data[error.start]:02x} on line {bad_line}",
                MAPPING,
                "save the file as UTF-8",
            )
            return None
        try:
            document = yaml.safe_load(text)
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
    """A YAML value's type, as an error names it.

    Args:
        value: A parsed YAML value.

    Returns:
        E.g. `list` or `string`.
    """
    names: dict[type, str] = {list: "list", str: "string"}
    return names.get(type(value), type(value).__name__)
