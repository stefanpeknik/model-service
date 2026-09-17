from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from workflow_execution.contracts import JsonValue


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class WorkflowInputDefinition:
    kinds: tuple[Literal["inline", "shared_path", "provider"], ...]
    media_type: str | None = None


@dataclass(frozen=True, slots=True)
class WorkflowValueDefinition:
    kind: Literal["inline", "shared_artifact"]
    media_type: str | None = None


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    identifier: str
    inputs: dict[str, WorkflowInputDefinition]
    outputs: dict[str, WorkflowValueDefinition]
    metadata: dict[str, object] = field(default_factory=dict)


class UnknownWorkflow(KeyError):  # noqa: N818 - Contract name from the design.
    pass


class MissingWorkflowApplicationsError(RuntimeError):
    def __init__(self, workflow_ids: tuple[str, ...]) -> None:
        super().__init__(
            "declared workflows are missing Ray Serve applications: "
            + ", ".join(workflow_ids)
        )
        self.workflow_ids = workflow_ids


class WorkflowCatalog:
    def __init__(
        self,
        definitions: tuple[WorkflowDefinition, ...],
        *,
        configuration_validators: Mapping[
            str,
            Callable[[Mapping[str, JsonValue]], object],
        ]
        | None = None,
    ) -> None:
        by_id = {definition.identifier: definition for definition in definitions}
        if len(by_id) != len(definitions):
            raise ValueError("workflow identifiers must be unique")
        self._definitions = by_id
        self._configuration_validators = dict(configuration_validators or {})

    @classmethod
    def default(cls) -> WorkflowCatalog:
        from workflow_execution.prostate_heatmap import validate_configuration

        return cls.from_directory(
            ROOT / "workflow-definitions",
            configuration_validators={
                "prostate-heatmap": validate_configuration,
            },
        )

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        *,
        configuration_validators: Mapping[
            str,
            Callable[[Mapping[str, JsonValue]], object],
        ]
        | None = None,
    ) -> WorkflowCatalog:
        definitions = tuple(
            _load_definition(path) for path in sorted(directory.glob("*.yaml"))
        )
        if not definitions:
            raise ValueError(f"no workflow definitions found in {directory}")
        return cls(
            definitions,
            configuration_validators=configuration_validators,
        )

    def get(self, workflow_id: str) -> WorkflowDefinition:
        try:
            return self._definitions[workflow_id]
        except KeyError as error:
            raise UnknownWorkflow(workflow_id) from error

    @property
    def identifiers(self) -> frozenset[str]:
        return frozenset(self._definitions)

    def validate_configuration(
        self,
        workflow_id: str,
        configuration: Mapping[str, JsonValue],
    ) -> None:
        validator = self._configuration_validators.get(workflow_id)
        if validator is not None:
            validator(configuration)

    def validate_deployed(self, deployed_workflow_ids: set[str]) -> None:
        missing = tuple(sorted(self._definitions.keys() - deployed_workflow_ids))
        if missing:
            raise MissingWorkflowApplicationsError(missing)

    def validate_helm_applications(
        self,
        values_path: Path,
        applications_directory: Path,
    ) -> None:
        application_ids = _load_enabled_application_ids(values_path)
        self.validate_deployed(application_ids)
        missing = tuple(
            sorted(
                workflow_id
                for workflow_id in self._definitions
                if _load_application_name(
                    applications_directory / f"{workflow_id}.yaml"
                )
                != workflow_id
            )
        )
        if missing:
            raise MissingWorkflowApplicationsError(missing)


def _load_definition(path: Path) -> WorkflowDefinition:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load workflow definition {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    identifier = value.get("identifier")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError(f"{path} needs a non-empty identifier")
    inputs = _load_inputs(value.get("inputs"), path)
    outputs = _load_outputs(value.get("outputs"), path)
    if set(inputs) & set(outputs):
        raise ValueError(f"{path} input and output names must not overlap")
    metadata = {
        key: item
        for key, item in value.items()
        if key not in {"identifier", "inputs", "outputs"}
    }
    return WorkflowDefinition(identifier, inputs, outputs, metadata)


def _load_enabled_application_ids(path: Path) -> set[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot load Helm values {path}") from error
    try:
        start = lines.index("applications:") + 1
    except ValueError as error:
        raise ValueError(f"{path} has no top-level applications list") from error
    application_ids: set[str] = set()
    for line in lines[start:]:
        if line and not line[0].isspace():
            break
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("- ") or not stripped[2:]:
            raise ValueError(f"{path} has an invalid applications entry")
        application_ids.add(stripped[2:])
    return application_ids


def _load_application_name(path: Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("- name: "):
            return line.removeprefix("- name: ").strip() or None
        if line.strip():
            return None
    return None


def _load_inputs(
    value: object,
    path: Path,
) -> dict[str, WorkflowInputDefinition]:
    definitions: dict[str, WorkflowInputDefinition] = {}
    for name, item, media_type in _load_entries(value, path, "inputs"):
        raw_kinds = item.get("kinds")
        allowed_kinds = {"inline", "shared_path", "provider"}
        if (
            not isinstance(raw_kinds, list)
            or not raw_kinds
            or any(kind not in allowed_kinds for kind in raw_kinds)
            or len(set(raw_kinds)) != len(raw_kinds)
        ):
            raise ValueError(f"{path} inputs.{name} has invalid kinds")
        definitions[name] = WorkflowInputDefinition(tuple(raw_kinds), media_type)
    return definitions


def _load_outputs(value: object, path: Path) -> dict[str, WorkflowValueDefinition]:
    definitions: dict[str, WorkflowValueDefinition] = {}
    for name, item, media_type in _load_entries(value, path, "outputs"):
        kind = item.get("kind")
        if kind not in {"inline", "shared_artifact"}:
            raise ValueError(f"{path} outputs.{name} has an invalid kind")
        definitions[name] = WorkflowValueDefinition(kind, media_type)
    return definitions


def _load_entries(
    value: object,
    path: Path,
    field_name: str,
) -> list[tuple[str, dict[str, object], str | None]]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} {field_name} must be an object")
    entries = []
    for name, item in value.items():
        if not isinstance(name, str) or not name or not isinstance(item, dict):
            raise ValueError(f"{path} has an invalid {field_name} entry")
        media_type = item.get("media_type")
        if media_type is not None and not isinstance(media_type, str):
            raise ValueError(f"{path} {field_name}.{name} has an invalid media type")
        entries.append((name, item, media_type))
    return entries
