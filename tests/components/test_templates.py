"""Each kind's template in `docs/guides/templates/`, copied and named, conforms."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from model_service.components import KINDS, Component
from model_service.testing import conformance


TEMPLATES = Path(__file__).resolve().parents[2] / "docs" / "guides" / "templates"


def copied(template: Path, directory: Path) -> type[Component]:
    """Copy a template with a name filled in, import it and return its component."""
    name = f"copied_{template.stem}"
    text = (
        template.read_text()
        .replace("my_component", name)
        .replace("MyComponent", f"Copied{template.stem.title()}")
    )
    path = directory / f"{name}.py"
    path.write_text(text)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return next(
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, Component)
        and value.__module__ == name
        and "name" in value.__dict__
    )


def test_every_kind_has_one_template() -> None:
    assert {t.stem for t in TEMPLATES.glob("*.py")} == {k.kind for k in KINDS}


@pytest.mark.parametrize(
    "template", sorted(TEMPLATES.glob("*.py")), ids=lambda t: t.stem
)
def test_a_copied_template_conforms(template: Path, tmp_path: Path) -> None:
    component = copied(template, tmp_path)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    assert component.kind == template.stem
    assert component.name == f"copied_{template.stem}"
    assert conformance.check(component, work_dir) == []
