from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml  # type: ignore[import-untyped]


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from workflow_execution.contracts import JsonValue


ROOT = Path(__file__).resolve().parents[1]

type WorkflowInputKind = Literal["inline", "shared_path", "provider"]
type WorkflowOutputKind = Literal["inline", "shared_artifact"]


@dataclass(frozen=True, slots=True)
class WorkflowInputDefinition:
    kinds: tuple[WorkflowInputKind, ...]
    media_type: str | None = None


@dataclass(frozen=True, slots=True)
class WorkflowValueDefinition:
    kind: WorkflowOutputKind
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
    value = _load_yaml(path)
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
    return WorkflowDefinition(
        identifier,
        inputs,
        outputs,
        metadata,
    )


def _load_enabled_application_ids(path: Path) -> set[str]:
    values = _load_yaml(path)
    if not isinstance(values, dict):
        raise TypeError(f"{path} must contain an object")
    applications = values.get("applications")
    if not isinstance(applications, list) or any(
        not isinstance(application, str) or not application
        for application in applications
    ):
        raise ValueError(f"{path} has an invalid applications list")
    return set(applications)


def _load_application_name(path: Path) -> str | None:
    try:
        application = _load_yaml(path)
    except ValueError:
        return None
    if not isinstance(application, list) or not application:
        return None
    first = application[0]
    if not isinstance(first, dict):
        return None
    name = first.get("name")
    return name if isinstance(name, str) and name else None


def _load_yaml(path: Path) -> object:
    try:
        value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot load YAML document {path}") from error
    return value


def _load_inputs(
    value: object,
    path: Path,
) -> dict[str, WorkflowInputDefinition]:
    definitions: dict[str, WorkflowInputDefinition] = {}
    for name, item, media_type in _load_entries(value, path, "inputs"):
        raw_kinds = item.get("kinds")
        if not isinstance(raw_kinds, list) or not raw_kinds:
            raise ValueError(f"{path} inputs.{name} has invalid kinds")
        try:
            kinds = tuple(_input_kind(kind) for kind in raw_kinds)
        except ValueError as error:
            raise ValueError(f"{path} inputs.{name} has invalid kinds") from error
        if len(set(kinds)) != len(kinds):
            raise ValueError(f"{path} inputs.{name} has duplicate kinds")
        definitions[name] = WorkflowInputDefinition(kinds, media_type)
    return definitions


def _load_outputs(value: object, path: Path) -> dict[str, WorkflowValueDefinition]:
    definitions: dict[str, WorkflowValueDefinition] = {}
    for name, item, media_type in _load_entries(value, path, "outputs"):
        try:
            kind = _output_kind(item.get("kind"))
        except ValueError as error:
            raise ValueError(f"{path} outputs.{name} has an invalid kind") from error
        definitions[name] = WorkflowValueDefinition(kind, media_type)
    return definitions


def _input_kind(value: object) -> WorkflowInputKind:
    if value == "inline":
        return "inline"
    if value == "shared_path":
        return "shared_path"
    if value == "provider":
        return "provider"
    raise ValueError("unknown workflow input kind")


def _output_kind(value: object) -> WorkflowOutputKind:
    if value == "inline":
        return "inline"
    if value == "shared_artifact":
        return "shared_artifact"
    raise ValueError("unknown workflow output kind")


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
