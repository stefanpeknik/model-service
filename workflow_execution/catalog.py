from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated

import yaml
from pydantic import ConfigDict, Field, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass as validated_dataclass

from workflow_execution.contracts import (  # noqa: TC001 - Pydantic resolves enum fields at runtime.
    WorkflowInputKind,
    WorkflowOutputKind,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path
    from typing import Self

    from workflow_execution.contracts import JsonValue


type Identifier = Annotated[str, Field(strict=True, min_length=1)]
type ConfigurationValidator = Callable[[Mapping[str, JsonValue]], object]
type MediaType = Annotated[str, Field(strict=True)] | None


@validated_dataclass(frozen=True, slots=True, config=ConfigDict(extra="ignore"))
class WorkflowInputDefinition:
    kinds: Annotated[tuple[WorkflowInputKind, ...], Field(min_length=1)]
    media_type: MediaType = None

    @model_validator(mode="after")
    def unique_kinds(self) -> Self:
        if len(set(self.kinds)) != len(self.kinds):
            raise ValueError("input kinds must be unique")
        return self


@validated_dataclass(frozen=True, slots=True, config=ConfigDict(extra="ignore"))
class WorkflowValueDefinition:
    kind: WorkflowOutputKind
    media_type: MediaType = None


@validated_dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    identifier: Identifier
    inputs: dict[Identifier, WorkflowInputDefinition]
    outputs: dict[Identifier, WorkflowValueDefinition]
    metadata: dict[str, object] = field(default_factory=dict)

    @model_validator(mode="after")
    def distinct_names(self) -> Self:
        if self.inputs.keys() & self.outputs.keys():
            raise ValueError("input and output names must not overlap")
        return self


@dataclass(frozen=True, slots=True)
class WorkflowConfiguration:
    validator: ConfigurationValidator
    defaults: Mapping[str, JsonValue] | None = None


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
        configurations: Mapping[str, WorkflowConfiguration],
    ) -> None:
        by_id = {definition.identifier: definition for definition in definitions}
        if len(by_id) != len(definitions):
            raise ValueError("workflow identifiers must be unique")
        self._definitions = by_id
        if configurations.keys() != by_id.keys():
            raise ValueError(
                "every declared workflow must have one configuration registration"
            )
        self._configurations = {
            workflow_id: WorkflowConfiguration(
                validator=configuration.validator,
                defaults=(
                    deepcopy(dict(configuration.defaults))
                    if configuration.defaults is not None
                    else None
                ),
            )
            for workflow_id, configuration in configurations.items()
        }
        for workflow_id, configuration in self._configurations.items():
            if configuration.defaults is not None:
                self.validate_configuration(workflow_id, configuration.defaults)

    @classmethod
    def default(cls) -> WorkflowCatalog:
        from workflow_execution.registry import default_catalog

        return default_catalog()

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        *,
        configurations: Mapping[str, WorkflowConfiguration],
    ) -> WorkflowCatalog:
        definitions = tuple(
            _load_definition(path) for path in sorted(directory.glob("*.yaml"))
        )
        if not definitions:
            raise ValueError(f"no workflow definitions found in {directory}")
        return cls(
            definitions,
            configurations=configurations,
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
        self, workflow_id: str, configuration: Mapping[str, JsonValue]
    ) -> None:
        self.get(workflow_id)
        self._configurations[workflow_id].validator(configuration)

    def resolve_configuration(
        self,
        workflow_id: str,
        configuration: Mapping[str, JsonValue],
        *,
        overrides: Mapping[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue]:
        self.get(workflow_id)
        resolved = deepcopy(dict(self._configurations[workflow_id].defaults or {}))
        resolved.update(deepcopy(dict(overrides or {})))
        resolved.update(deepcopy(dict(configuration)))
        self.validate_configuration(workflow_id, resolved)
        return resolved

    def validate_deployed(self, deployed_workflow_ids: set[str]) -> None:
        missing = tuple(sorted(self._definitions.keys() - deployed_workflow_ids))
        if missing:
            raise MissingWorkflowApplicationsError(missing)


def validate_empty_configuration(configuration: Mapping[str, JsonValue]) -> None:
    if configuration:
        raise ValueError("workflow configuration must be empty")


_DEFINITION = TypeAdapter(WorkflowDefinition)


def _load_definition(path: Path) -> WorkflowDefinition:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot load YAML document {path}") from error
    if isinstance(value, dict):
        fields = {"identifier", "inputs", "outputs"}
        value = {
            **{key: item for key, item in value.items() if key in fields},
            "metadata": {key: item for key, item in value.items() if key not in fields},
        }
    return _DEFINITION.validate_python(value)
