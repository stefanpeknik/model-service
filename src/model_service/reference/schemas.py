"""JSON Schema files for catalog files, and the line linking each file to its schema.

Served-model and analysis files each get `catalog/.schemas/<folder>.schema.json`
and, for environment patches, `<folder>.patch.schema.json`. Each catalog file
starts with a `# yaml-language-server: $schema=<relative path>` comment, which
editors with the YAML language server read; the loader and Helm ignore it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from model_service.catalog import analyses, served_models
from model_service.catalog.loader import FOLDERS, environments
from model_service.catalog.schema import file_schema, patch_schema


if TYPE_CHECKING:
    from model_service.registry import Registry


SCHEMAS = ".schemas"
MODELINE = "# yaml-language-server: $schema="


def schema_files(registry: Registry, catalog: Path) -> dict[Path, str]:
    """The schema of served-model and analysis files, and of their patches.

    Args:
        registry: The components files may name.
        catalog: The catalog directory, relative to the repo.

    Returns:
        File contents by path relative to the repo.
    """
    models = {
        served_models.FOLDER: served_models.served_model_schema(registry),
        analyses.FOLDER: analyses.analysis_schema(registry),
    }
    files = {}
    for folder, model in models.items():
        schema = file_schema(model)
        files[catalog / SCHEMAS / f"{folder}.schema.json"] = _json(schema)
        files[catalog / SCHEMAS / f"{folder}.patch.schema.json"] = _json(
            patch_schema(schema)
        )
    return files


def linked_files(repo: Path, catalog: Path, envs: Path) -> dict[Path, str]:
    """Every catalog file and environment patch, starting with the line that links its schema.

    Args:
        repo: The repository root.
        catalog: The catalog directory, relative to `repo`.
        envs: The folder of environments, relative to `repo`.

    Returns:
        File contents by path relative to the repo.
    """
    found: list[tuple[Path, str]] = []
    for folder in FOLDERS:
        found += [
            (path, f"{folder}.schema.json")
            for path in sorted((repo / catalog / folder).glob("*.yaml"))
        ]
        for environment in environments(repo / envs):
            found += [
                (path, f"{folder}.patch.schema.json")
                for path in sorted((environment / folder).glob("*.yaml"))
            ]
    files = {}
    for path, schema in found:
        up = PurePosixPath(os.path.relpath(repo / catalog / SCHEMAS, path.parent))
        files[path.relative_to(repo)] = with_modeline(
            path.read_text(), f"{MODELINE}{up / schema}"
        )
    return files


def with_modeline(text: str, line: str) -> str:
    """The file with `line` as its first line, replacing an older modeline.

    Args:
        text: The file's content.
        line: The modeline.

    Returns:
        The content starting with `line`.
    """
    first, _, rest = text.partition("\n")
    body = rest if first.startswith(MODELINE) else text
    return f"{line}\n{body}"


def _json(schema: dict[str, object]) -> str:
    """A schema as the JSON file holds it.

    Args:
        schema: The JSON schema.

    Returns:
        The schema, indented, with a final newline.
    """
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"
