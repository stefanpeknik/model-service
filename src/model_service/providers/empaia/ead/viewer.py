"""The AigoPath case browser's settings for one app, in its syntax version 2.

The case browser reads them from the app's global configuration key
`aigopath_viewer`. Each io key's entry holds `label` and `description` and one
block named after its kind (`number`, `annotation`, `pixelmap`, ...), which
the EAD type decides; `aigopath-viewer.schema.json` is the case browser's
schema, vendored unchanged, and says which types each kind covers
(`x-eadTypes`). A parameter's block gets its setting's default and inclusive
bounds, so the pipeline file states them once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonschema import Draft7Validator


if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import BaseModel

    from model_service.contracts.json_values import JsonValue
    from model_service.providers.empaia.ead.vocabulary import ViewerConfig


VERSION = 2
_SCHEMA = json.loads(
    Path(__file__).with_name("aigopath-viewer.schema.json").read_text()
)
VIEWER_SCHEMA = Draft7Validator(_SCHEMA)
COMMON = frozenset({"label", "description"})


def _kinds(entry: str) -> dict[str, str]:
    """The kind block of each EAD type, from the schema's `x-eadTypes`.

    Args:
        entry: `inputEntry` or `outputEntry`.

    Returns:
        EAD type to kind, e.g. `float` to `number`.
    """
    definitions = _SCHEMA["definitions"]
    found: dict[str, str] = {}
    for kind, value in definitions[entry]["properties"].items():
        if "$ref" in value:
            block = definitions[value["$ref"].rsplit("/", 1)[1]]
            found.update(dict.fromkeys(block["x-eadTypes"], kind))
    return found


INPUT_KINDS = _kinds("inputEntry")
OUTPUT_KINDS = _kinds("outputEntry")


def _ead_type(io_entry: Mapping[str, Any]) -> str:
    """The type an entry describes: a collection's item type, otherwise its own.

    Args:
        io_entry: The EAD io entry.

    Returns:
        The EAD type.
    """
    while io_entry["type"] == "collection":
        io_entry = io_entry["items"]
    return str(io_entry["type"])


def _kind(kinds: Mapping[str, str], io_entry: Mapping[str, Any]) -> str:
    """The kind block of an io entry; an unknown type stays as is, which the schema refuses.

    Args:
        kinds: `INPUT_KINDS` or `OUTPUT_KINDS`.
        io_entry: The EAD io entry.

    Returns:
        The kind, e.g. `number`.
    """
    ead_type = _ead_type(io_entry)
    return kinds.get(ead_type, ead_type)


def _parameter_block(setting: Mapping[str, Any], value: JsonValue) -> dict[str, Any]:
    """A parameter's default, inclusive bounds and choices, from its setting.

    Args:
        setting: The setting's JSON Schema.
        value: Its value in the pipeline file.

    Returns:
        `value`, and `min`, `max` or `options` where the setting has them.
    """
    block: dict[str, Any] = {"value": value}
    if "minimum" in setting:
        block["min"] = setting["minimum"]
    if "maximum" in setting:
        block["max"] = setting["maximum"]
    if "enum" in setting:
        block["options"] = [{"value": choice} for choice in setting["enum"]]
    return block


def _entry(
    settings: BaseModel | None, kind: str, block: dict[str, Any]
) -> dict[str, Any]:
    """One io key's entry: the common fields, and the rest in its kind block.

    Args:
        settings: The entry as written in the pipeline file, if it is there.
        kind: The kind block's name.
        block: What the kind block holds before the written settings.

    Returns:
        The entry.
    """
    written = settings.model_dump(exclude_none=True) if settings else {}
    common = {k: v for k, v in written.items() if k in COMMON}
    rest = {k: v for k, v in written.items() if k not in COMMON}
    return {**common, kind: {**block, **rest}}


def viewer_config(
    viewer: ViewerConfig,
    io: Mapping[str, Mapping[str, Any]],
    parameters: Mapping[str, tuple[Mapping[str, Any], JsonValue]],
) -> dict[str, Any]:
    """Build the case browser's settings for an app.

    Inputs get an entry when the pipeline file writes one or when they are
    parameters; outputs when the file writes one, in its order. Entries for
    io keys the app lacks are left out, e.g. the region of interest of an app
    that only preprocesses.

    Args:
        viewer: The pipeline file's `viewer:` settings.
        io: The app's EAD io entries.
        parameters: Each parameter's setting schema and value, by io key.

    Returns:
        The settings; `VIEWER_SCHEMA` validates them.
    """
    inputs: dict[str, Any] = {}
    for key in [*viewer.inputs, *(p for p in parameters if p not in viewer.inputs)]:
        if key not in io:
            continue
        setting, value = parameters.get(key, ({}, None))
        block = _parameter_block(setting, value) if key in parameters else {}
        kind = _kind(INPUT_KINDS, io[key])
        inputs[key] = _entry(viewer.inputs.get(key), kind, block)
    outputs = {
        key: _entry(entry, _kind(OUTPUT_KINDS, io[key]), {})
        for key, entry in viewer.outputs.items()
        if key in io
    }
    return {"version": VERSION, "inputs": inputs, "outputs": outputs}
