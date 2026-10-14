"""Generate an EMPAIA EAD from an app file binding a builder to a model.

An app file (`eads/apps/<app>.yaml`) names the builder, the model deployment and
the EMPAIA metadata. Input and output types come from the builder's EMPAIA
mappings in `builders/registry.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from jsonschema import Draft7Validator, validate

from builders.registry import get_builder


if TYPE_CHECKING:
    from builders.registry import BuilderSpec


EAD_SCHEMA = "https://gitlab.com/empaia/integration/definitions/-/raw/main/ead/ead-schema.v3.json"
ROOT = Path(__file__).resolve().parents[1]
_EAD_SCHEMA = json.loads(
    (ROOT / "eads" / "schemas" / "ead-v3.schema.json").read_text(encoding="utf-8")
)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return value


def generate_ead(
    app: dict[str, Any], spec: BuilderSpec | None = None
) -> dict[str, Any]:
    """Build and schema-validate the EAD; `spec` overrides the registered builder."""
    spec = spec or get_builder(app["builder"])
    if not isinstance(app.get("model"), str) or not app["model"]:
        raise ValueError("app must name a model deployment")
    empaia = app["empaia"]
    inputs = {name: item.empaia for name, item in spec.inputs.items() if item.empaia}
    outputs = {name: item.empaia for name, item in spec.outputs.items()}
    for name, mapping in outputs.items():
        if not mapping:
            raise ValueError(f"builder {spec.id!r} output {name!r} has no EMPAIA type")

    io = {}
    for name, mapping in {**inputs, **outputs}.items():
        entry = dict(mapping)
        if "reference" in entry:
            target = entry["reference"]
            if target not in inputs:
                raise ValueError(f"{name!r} references unknown input {target!r}")
            entry["reference"] = f"io.{target}"
        io[name] = entry

    ead = {
        "$schema": EAD_SCHEMA,
        "name": empaia["name"],
        "name_short": empaia["name_short"],
        "namespace": empaia["namespace"],
        "description": empaia["description"],
        "permissions": empaia["permissions"],
        "io": io,
        "modes": {empaia["mode"]: {"inputs": list(inputs), "outputs": list(outputs)}},
    }
    validate(ead, _EAD_SCHEMA, cls=Draft7Validator)
    return ead


def generate_file(app_path: Path, output_path: Path, *, check: bool = False) -> bool:
    """Write the EAD, or with `check` only compare it with the existing file."""
    rendered = json.dumps(
        generate_ead(load_yaml(app_path)), indent=2, ensure_ascii=False
    )
    rendered += "\n"
    if check:
        return (
            output_path.exists() and output_path.read_text(encoding="utf-8") == rendered
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not generate_file(args.app, args.output, check=args.check):
        parser.error(f"{args.output} is missing or out of date")


if __name__ == "__main__":
    main()
