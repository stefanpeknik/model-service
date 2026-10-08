"""Generate an EMPAIA EAD from a model-service workflow definition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft7Validator, validate


EAD_SCHEMA = "https://gitlab.com/empaia/integration/definitions/-/raw/main/ead/ead-schema.v3.json"
ROOT = Path(__file__).resolve().parents[1]
_EAD_SCHEMA = json.loads(
    (ROOT / "eads" / "schemas" / "ead-v3.schema.json").read_text(encoding="utf-8")
)


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a UTF-8 YAML document and require its top-level value to be a mapping.

    Args:
        path: File, storage locator, or HTTP endpoint selected by this operation.

    Returns:
        dict[str, Any]: Top-level YAML mapping with its workflow and EMPAIA fields.

    Raises:
        TypeError: A value or configured adapter has an unsupported type.
    """
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return value


def generate_ead(definition: dict[str, Any]) -> dict[str, Any]:
    """Translate workflow metadata and I/O into an EAD and validate its pinned schema.

    Args:
        definition: Workflow input/output contract or loaded YAML mapping to translate.

    Returns:
        dict[str, Any]: EAD schema/name/permission metadata, translated io references,
            and one declared execution mode.

    Raises:
        ValueError: Input/output names overlap or an I/O reference targets a missing input.
        jsonschema.ValidationError: The generated EAD violates the pinned EMPAIA schema.
    """
    empaia = definition["empaia"]
    inputs = definition["inputs"]
    outputs = definition["outputs"]
    if set(inputs) & set(outputs):
        raise ValueError("workflow input and output names must not overlap")

    io = {}
    for name, item in {**inputs, **outputs}.items():
        entry = dict(item["empaia"])
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
        "modes": {
            empaia["mode"]: {
                "inputs": list(inputs),
                "outputs": list(outputs),
            }
        },
    }
    validate(ead, _EAD_SCHEMA, cls=Draft7Validator)
    return ead


def generate_file(
    definition_path: Path, output_path: Path, *, check: bool = False
) -> bool:
    """Render deterministic EAD JSON and write it, or in check mode compare it with the existing file without modifying it.

    Args:
        definition_path: Source YAML workflow definition file.
        output_path: Destination artifact path or configured relative output path.
        check: Whether to compare generated metadata without writing the output file.

    Returns:
        bool: True after writing, or when check mode finds identical generated content;
            False for missing/stale checked output.
    """
    definition = load_yaml(definition_path)
    ead = generate_ead(definition)
    rendered = json.dumps(ead, indent=2, ensure_ascii=False) + "\n"
    if check:
        return (
            output_path.exists() and output_path.read_text(encoding="utf-8") == rendered
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return True


def main() -> None:
    """Parse definition/output paths and optional check mode; report a CLI error if generated metadata is missing or stale.

    Returns:
        None: No return value.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("definition", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not generate_file(args.definition, args.output, check=args.check):
        parser.error(f"{args.output} is missing or out of date")


if __name__ == "__main__":
    main()
