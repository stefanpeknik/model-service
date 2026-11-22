"""A resolved analysis: its components, models, inputs, settings and columns."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

from model_service.catalog.analyses.model_step import ModelStep
from model_service.components import per_run_fields, with_overrides


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.catalog.served_models import ServedModel
    from model_service.components import Col, Component, Sink, Source, Step


type AnalysisStep = Step | ModelStep


@dataclass(frozen=True)
class Analysis:
    """One analysis file, resolved and checked.

    Attributes:
        id: The analysis ID, the file name.
        summary: One line saying what it does.
        source: The source, with its tile geometry filled in.
        steps: Component steps and model steps, in order.
        sinks: The sinks.
        inputs: Input name to whether a run must give it; the union of what
            the components read.
        columns: The columns that reach the sinks, with every dimension known.
    """

    id: str
    summary: str
    source: Source
    steps: tuple[AnalysisStep, ...]
    sinks: tuple[Sink, ...]
    inputs: Mapping[str, bool]
    columns: Mapping[str, Col]

    @property
    def models(self) -> dict[str, ServedModel]:
        """The served models the model steps call, by ID, in step order."""
        return {s.model.id: s.model for s in self.steps if isinstance(s, ModelStep)}

    @property
    def last_model(self) -> ServedModel | None:
        """The last model step's served model, which sinks are told about."""
        models = list(self.models.values())
        return models[-1] if models else None

    def settings(self) -> dict[str, Any]:
        """The per-run settings and their values from the file.

        Returns:
            Setting name to value: the `PerRun` options of every component,
            under their own names.
        """
        return {
            name: getattr(component.options, name)
            for component in self._components()
            for name in sorted(per_run_fields(type(component.options)))
        }

    def with_settings(self, values: Mapping[str, Any]) -> Analysis:
        """The analysis with a caller's per-run settings applied.

        Args:
            values: Setting name to value.

        Returns:
            A new analysis; this one is unchanged.

        Raises:
            ValueError: For a name that is not a setting.
            pydantic.ValidationError: For an invalid value.
        """
        unknown = sorted(set(values) - set(self.settings()))
        if unknown:
            raise ValueError(
                f"{', '.join(unknown)}: not settings of {self.id}; settings: "
                f"{', '.join(sorted(self.settings())) or 'none'}"
            )

        def tuned[C: Component](component: C) -> C:
            mine = {
                k: v
                for k, v in values.items()
                if k in per_run_fields(type(component.options))
            }
            if not mine:
                return component
            return type(component)(with_overrides(component.options, mine))

        return replace(
            self,
            source=tuned(self.source),
            steps=tuple(
                s if isinstance(s, ModelStep) else tuned(s) for s in self.steps
            ),
            sinks=tuple(tuned(s) for s in self.sinks),
        )

    def _components(self) -> list[Component]:
        steps = [
            cast("Component", s) for s in self.steps if not isinstance(s, ModelStep)
        ]
        return [self.source, *steps, *self.sinks]
