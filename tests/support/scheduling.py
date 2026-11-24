"""Scheduler doubles shared by scheduler tests: a minimal provider, pipelines and a clock."""

from __future__ import annotations

import functools
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from model_service.catalog import load_catalog
from model_service.catalog.pipelines import resolve_run
from model_service.contracts.invocation import (
    InputAccessContext,
    Invocation,
    SharedPathInput,
)
from model_service.control_plane.scheduler.public import (
    PreparedInvocation,
    SubmissionIdentity,
)
from model_service.control_plane.scheduler.targets import ExecutionTarget
from model_service.providers.provider import ProviderAdapter
from tests.support.pipeline_catalog import REGISTRY, model_file, write


if TYPE_CHECKING:
    from datetime import datetime, timedelta

    from model_service.catalog import Catalog
    from model_service.catalog.pipelines import ResolvedPipeline
    from model_service.contracts.json_values import JsonObject, JsonValue
    from model_service.control_plane.scheduler.public import (
        BeforeMutation,
        PipelineScheduleRequest,
        ProviderFinishOutcome,
    )


# Two pipelines on two models: `a1` calls `m1`, `a2` calls `m2`.
PIPELINES = {"a1": "m1", "a2": "m2"}


def stored_target(pipeline_id: str, models: tuple[str, ...] = ()) -> ExecutionTarget:
    """A target as admission stores it, with a stand-in document and digest."""
    document: JsonObject = {
        "id": pipeline_id,
        "steps": [{"model": model} for model in models],
        "sinks": [],
    }
    return ExecutionTarget(pipeline_id, models, f"digest-{pipeline_id}", document)


@functools.cache
def scheduling_catalog() -> Catalog:
    """The catalog the scheduler tests resolve pipelines in."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for pipeline_id, model_id in PIPELINES.items():
            write(root, f"models/{model_id}.yaml", model_file())
            write(
                root,
                f"pipelines/{pipeline_id}.yaml",
                f"""
                source: {{use: slide_tiles}}
                steps: [{{model: {model_id}}}]
                sinks: [{{use: score_grid, output: heatmap}}]
                """,
            )
        return load_catalog(root, registry=REGISTRY)


def resolve_test_pipeline(
    pipeline_id: str, settings: JsonObject | None
) -> ResolvedPipeline:
    """Resolve one of `PIPELINES` as the control plane's resolver would."""
    return resolve_run(scheduling_catalog(), pipeline_id, settings)


class StandaloneProvider(ProviderAdapter):
    """Prepare minimal scheduler invocations without shared-file checks so tests can isolate orchestration."""

    name = "standalone"

    def identify(self, request: PipelineScheduleRequest) -> SubmissionIdentity:
        """Use the test replay key with a fixed fingerprint to isolate scheduler replay behavior."""
        assert request.idempotency_key is not None
        return SubmissionIdentity(
            provider=self.name,
            key=request.idempotency_key,
            fingerprint="same-request",
        )

    async def prepare(
        self,
        request: PipelineScheduleRequest,
        deadline_at: datetime,
    ) -> PreparedInvocation:
        """Create a shared-storage invocation of a slide with the scheduler-supplied deadline."""
        return PreparedInvocation(
            invocation=Invocation(
                inputs={"slide": SharedPathInput("slide.tiff")},
                configuration={},
                input_access=InputAccessContext(
                    kind="shared_storage",
                    state={"root": "/shared"},
                ),
                deadline_at=deadline_at,
            ),
            protected_context={},
        )

    async def finish(
        self,
        protected_context: dict[str, JsonValue],
        outcome: ProviderFinishOutcome,
        *,
        before_mutation: BeforeMutation,
    ) -> None:
        """Complete without external writes or mutation guarding for standalone orchestration tests."""
        del protected_context, outcome, before_mutation


class MutableClock:
    """Provide a manually advanced timestamp for scheduler lease, retry, and deadline tests."""

    def __init__(self, now: datetime) -> None:
        """Set the initial aware timestamp supplied by the test."""
        self.now = now

    def __call__(self) -> datetime:
        """Return the current test timestamp without advancing it."""
        return self.now

    def advance(self, duration: timedelta) -> None:
        """Advance simulated time by the supplied duration without sleeping."""
        self.now += duration
