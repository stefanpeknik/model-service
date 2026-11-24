"""Generated files: catalog JSON Schemas and the reference docs.

`generated(repo)` computes every generated file from the component classes,
the format modules and `catalog/`; `tools/generate.py` writes them and a
test fails when the committed copies differ. The generated files are:

- `catalog/.schemas/<folder>.schema.json` and `<folder>.patch.schema.json`
  for served-model and analysis files;
- the `# yaml-language-server: $schema=…` first line of each catalog file;
- `docs/reference/components.md`, `served-models.md`, `analyses.md` and
  `formats.md`.
"""

from __future__ import annotations

from pathlib import Path

from model_service.catalog import load_catalog
from model_service.reference.analyses_page import analyses_page
from model_service.reference.components_page import components_page
from model_service.reference.formats_page import formats_page
from model_service.reference.schemas import linked_files, schema_files
from model_service.reference.served_models_page import served_models_page
from model_service.registry import default_registry


CATALOG = Path("catalog")
ENVIRONMENTS = Path("deploy/helm/environments")
REFERENCE = Path("docs/reference")


def generated(repo: Path) -> dict[Path, str]:
    """Every generated file's content.

    Args:
        repo: The repository root.

    Returns:
        Contents by path relative to `repo`.

    Raises:
        CatalogInvalid: If the base catalog does not load.
    """
    registry = default_registry()
    catalog = load_catalog(repo / CATALOG)
    return {
        **schema_files(registry, CATALOG),
        **linked_files(repo, CATALOG, ENVIRONMENTS),
        REFERENCE / "components.md": components_page(registry, catalog),
        REFERENCE / "served-models.md": served_models_page(
            catalog, repo / ENVIRONMENTS
        ),
        REFERENCE / "analyses.md": analyses_page(catalog),
        REFERENCE / "formats.md": formats_page(registry),
    }


def stale(repo: Path) -> list[Path]:
    """The generated files whose committed copy differs, and leftovers.

    A leftover is a file in `catalog/.schemas/` or `docs/reference/` that
    generation no longer writes.

    Args:
        repo: The repository root.

    Returns:
        Paths relative to `repo`, sorted.
    """
    return _stale(repo, generated(repo))


def write(repo: Path) -> list[Path]:
    """Write every generated file that changed and remove leftovers.

    Args:
        repo: The repository root.

    Returns:
        The paths written or removed, relative to `repo`.
    """
    files = generated(repo)
    changed = _stale(repo, files)
    for path in changed:
        target = repo / path
        if path in files:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(files[path])
        else:
            target.unlink()
    return changed


def _stale(repo: Path, files: dict[Path, str]) -> list[Path]:
    """The generated files that differ from the repository, or that it holds but generation no longer writes.

    Args:
        repo: The repository root.
        files: What generation writes, by path relative to the root.

    Returns:
        The stale paths, relative to the root, sorted.
    """
    differ = {
        path
        for path, text in files.items()
        if not (repo / path).is_file() or (repo / path).read_text() != text
    }
    leftovers = {
        path.relative_to(repo)
        for folder in (CATALOG / ".schemas", REFERENCE)
        for path in (repo / folder).glob("*")
        if path.relative_to(repo) not in files
    }
    return sorted(differ | leftovers)
