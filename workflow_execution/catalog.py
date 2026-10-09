"""Load workflow definitions and resolve validated configuration independently of deployment.

YAML declares named inputs and outputs; registrations attach validators and defaults.
Resolved configuration is copied so executions do not share mutable caller state.
"""

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
    """Declare accepted input kinds and an optional media type for one named input.

    Attributes:
        kinds: Distinct input representations accepted by the declaration.
        media_type: Optional declared artifact content type, such as image/tiff.
    """

    kinds: Annotated[tuple[WorkflowInputKind, ...], Field(min_length=1)]
    media_type: MediaType = None

    @model_validator(mode="after")
    def unique_kinds(self) -> Self:
        """Reject repeated accepted kinds in a workflow input definition.

        Returns:
            Self: The same validated definition instance.

        Raises:
            ValueError: kinds contains duplicate input representations.
        """
        if len(set(self.kinds)) != len(self.kinds):
            raise ValueError("input kinds must be unique")
        return self


@validated_dataclass(frozen=True, slots=True, config=ConfigDict(extra="ignore"))
class WorkflowValueDefinition:
    """Declare one output's kind and optional artifact media type.

    Attributes:
        kind: Tagged input/output representation or access-adapter selector.
        media_type: Optional declared artifact content type, such as image/tiff.
    """

    kind: WorkflowOutputKind
    media_type: MediaType = None


@validated_dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    """Describe a workflow's identifier, named inputs/outputs, and retained descriptive YAML metadata.

    Attributes:
        identifier: Unique workflow routing identifier.
        inputs: Input names mapped to their accepted representation definitions.
        outputs: Output names mapped to their required representation definitions.
        metadata: Descriptive YAML fields outside the validated workflow contract.
    """

    identifier: Identifier
    inputs: dict[Identifier, WorkflowInputDefinition]
    outputs: dict[Identifier, WorkflowValueDefinition]
    metadata: dict[str, object] = field(default_factory=dict)

    @model_validator(mode="after")
    def distinct_names(self) -> Self:
        """Reject a name used for both an input and an output.

        Returns:
            Self: The same validated definition instance.

        Raises:
            ValueError: An input name is also declared as an output name.
        """
        if self.inputs.keys() & self.outputs.keys():
            raise ValueError("input and output names must not overlap")
        return self


@dataclass(frozen=True, slots=True)
class WorkflowConfiguration:
    """Pair a workflow's configuration validator with optional default values.

    Attributes:
        validator: Callable validating resolved workflow configuration.
        defaults: Optional default JSON values copied before caller overrides.
    """

    validator: ConfigurationValidator
    defaults: Mapping[str, JsonValue] | None = None


class UnknownWorkflow(KeyError):  # noqa: N818 - Contract name from the design.
    """Signal lookup of a workflow identifier absent from the catalog."""


class WorkflowCatalog:
    """Index definitions and their configuration registrations for admission and execution validation."""

    def __init__(
        self,
        definitions: tuple[WorkflowDefinition, ...],
        *,
        configurations: Mapping[str, WorkflowConfiguration],
    ) -> None:
        """Require unique identifiers and one configuration registration per definition; copy and validate defaults.

        Args:
            definitions: Declared workflow contracts to index by unique identifier.
            configurations: One validator/default registration per declared workflow ID.

        Raises:
            ValueError: Workflow identifiers repeat, configuration registrations do not
                match the declared workflows, or defaults fail their validator.
            TypeError: A configuration validator rejects the type of a default value.
        """
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
    def default(cls, identifiers: frozenset[str] | None = None) -> WorkflowCatalog:
        """Load the built-in definitions and validators from workflows.registry.

        Args:
            identifiers: Optional nonempty built-in selection; None loads all workflows.

        Returns:
            WorkflowCatalog: Indexed workflow contracts and configuration registrations.

        Raises:
            ValueError: The requested selection is empty.
            UnknownWorkflow: A requested identifier has no built-in declaration.
        """
        from workflows.registry import default_catalog

        catalog = default_catalog()
        return catalog if identifiers is None else catalog.select(identifiers)

    def select(self, identifiers: frozenset[str]) -> WorkflowCatalog:
        """Restrict declarations and configuration registrations to deployed workflows.

        Args:
            identifiers: Nonempty set of known workflow IDs enabled in this deployment.

        Returns:
            WorkflowCatalog: Independent catalog containing only selected workflows.

        Raises:
            ValueError: Selection is empty.
            UnknownWorkflow: A selected identifier is not a built-in declaration.
        """
        if not identifiers:
            raise ValueError("at least one workflow must be enabled")
        return WorkflowCatalog(
            tuple(self.get(identifier) for identifier in sorted(identifiers)),
            configurations={
                identifier: self._configurations[identifier]
                for identifier in identifiers
            },
        )

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        *,
        configurations: Mapping[str, WorkflowConfiguration],
    ) -> WorkflowCatalog:
        """Read sorted YAML definitions from a directory, requiring at least one, and attach configuration registrations.

        Args:
            directory: Directory containing definitions or generated validation
                fixtures.
            configurations: One validator/default registration per declared workflow ID.

        Returns:
            WorkflowCatalog: Indexed workflow contracts and configuration registrations.

        Raises:
            ValueError: No YAML definitions exist, a file cannot be read/parsed,
                identifiers repeat, registrations do not match, or defaults are invalid.
            ValidationError: A parsed YAML definition violates the typed workflow
                schema.
            TypeError: A registered configuration validator rejects a default value
                type.
        """
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
        """Return a declared workflow or raise UnknownWorkflow for an absent identifier.

        Args:
            workflow_id: Declared workflow identifier used for catalog lookup and
                dispatch routing.

        Returns:
            WorkflowDefinition: Validated named input/output contract for the selected
                workflow.

        Raises:
            UnknownWorkflow: The workflow identifier is absent from the catalog.
        """
        try:
            return self._definitions[workflow_id]
        except KeyError as error:
            raise UnknownWorkflow(workflow_id) from error

    @property
    def identifiers(self) -> frozenset[str]:
        """Return an immutable set of declared workflow IDs for scheduler routing and health checks.

        Returns:
            frozenset[str]: All declared workflow identifiers as an immutable routing
                set.
        """
        return frozenset(self._definitions)

    def validate_configuration(
        self, workflow_id: str, configuration: Mapping[str, JsonValue]
    ) -> None:
        """Run the registered validator for a known workflow without changing its configuration.

        Args:
            workflow_id: Declared workflow identifier used for catalog lookup and
                dispatch routing.
            configuration: Resolved workflow values or provider global/customer sections
                to validate.

        Returns:
            None: No return value.

        Raises:
            UnknownWorkflow: workflow_id is absent from the catalog.
            ValueError: The registered validator rejects a configuration value or field.
            TypeError: The registered validator rejects a configuration value type.
        """
        self.get(workflow_id)
        self._configurations[workflow_id].validator(configuration)

    def resolve_configuration(
        self,
        workflow_id: str,
        configuration: Mapping[str, JsonValue],
    ) -> dict[str, JsonValue]:
        """Copy defaults, shallowly override them with copied caller values, and validate the resulting configuration.

        Args:
            workflow_id: Declared workflow identifier used for catalog lookup and
                dispatch routing.
            configuration: Resolved workflow values or provider global/customer sections
                to validate.

        Returns:
            dict[str, JsonValue]: A separate validated copy of defaults shallowly
                overridden by caller values.

        Raises:
            UnknownWorkflow: workflow_id is absent from the catalog.
            ValueError: The merged configuration fails its registered validator.
            TypeError: The merged configuration contains a value type rejected by its
                validator.
        """
        self.get(workflow_id)
        resolved = deepcopy(dict(self._configurations[workflow_id].defaults or {}))
        resolved.update(deepcopy(dict(configuration)))
        self.validate_configuration(workflow_id, resolved)
        return resolved


def validate_empty_configuration(configuration: Mapping[str, JsonValue]) -> None:
    """Reject any configuration values for workflows that declare no configurable behavior.

    Args:
        configuration: Resolved workflow values or provider global/customer sections to
            validate.

    Returns:
        None: No return value.

    Raises:
        ValueError: configuration contains any fields.
    """
    if configuration:
        raise ValueError("workflow configuration must be empty")


_DEFINITION = TypeAdapter(WorkflowDefinition)


def _load_definition(path: Path) -> WorkflowDefinition:
    """Read and validate YAML, separating contract fields from descriptive metadata; translate file/YAML errors into ValueError.

    Args:
        path: UTF-8 YAML file declaring one workflow contract and descriptive metadata.

    Returns:
        WorkflowDefinition: Validated workflow contract with non-contract YAML fields
            retained as metadata.

    Raises:
        ValueError: path cannot be read or parsed as YAML.
        ValidationError: The parsed document violates the typed workflow definition
            schema.
    """
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
