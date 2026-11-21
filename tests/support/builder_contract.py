"""What every builder must do, run against a real slide and a fake model.

A builder's tests subclass `BuilderContractTests` with
`unittest.IsolatedAsyncioTestCase`, naming the builder and a model of its
contract; the builder is loaded from its declaration as a Ray replica loads it.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from model_service.builders.base import ReplicaSettings
from model_service.builders.input_access import InputAccess, InputAccessFactory
from model_service.builders.registry import get_builder
from model_service.builders.replica.runner import BuilderApplication
from model_service.contracts.failures import (
    BuilderFailure,
    FailureCode,
    InvalidInvocation,
)
from model_service.contracts.invocation import SharedPathInput
from model_service.contracts.model_declarations import InferenceContract
from model_service.contracts.result import SharedArtifact
from model_service.providers.standalone.shared_storage import SharedStorageAccessFactory
from model_service.testing import fake_slide
from tests.support.builder_fixtures import (
    FakeDeployment,
    invocation,
    running_config,
)


if TYPE_CHECKING:
    import unittest

    from model_service.contracts.invocation import (
        BuilderInput,
        ExecutionContext,
        Invocation,
    )
    from model_service.contracts.model_declarations import ModelBinding
    from model_service.contracts.result import BuilderResult

    _Base = unittest.IsolatedAsyncioTestCase
else:
    _Base = object


def healthy_deployment(model: ModelBinding) -> FakeDeployment:
    """A deployment running as declared that answers every tile within the contract."""
    shape: tuple[int, ...]
    if model.contract is InferenceContract.SPATIAL_SCORES:
        shape = (model.n_channels, model.output_tile_size, model.output_tile_size)
        return FakeDeployment(
            running_config(model),
            predict=lambda _tile: np.full(shape, 0.5, dtype=np.float32),
        )
    summaries = len(model.summaries)
    shape = (4,) if summaries == 1 else (summaries, 4)
    return FakeDeployment(
        running_config(model),
        embed=lambda _tile: np.ones(shape, dtype=np.float32),
    )


class BuilderContractTests(_Base):
    builder_id: ClassVar[str]
    # Configuration every contract run uses, e.g. to turn off tissue detection.
    configuration: ClassVar[dict[str, Any]] = {}
    # A model of the builder's contract, at 0.5 mpp so the slide is read at level 0.
    model: ClassVar[ModelBinding]

    async def asyncSetUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        # H&E-like pink, so a background filter sees tissue everywhere.
        pixels = np.empty((64, 96, 3), dtype=np.uint8)
        pixels[...] = (200, 120, 160)
        fake_slide(self.root / "slide.tiff", pixels)
        self.spec = get_builder(self.builder_id)

    def inputs(self) -> dict[str, BuilderInput]:
        """The invocation's inputs; override for a builder that needs more than a slide."""
        return {"slide": SharedPathInput("slide.tiff")}

    async def run_builder(
        self,
        deployment: FakeDeployment | None = None,
        *,
        progress: list[float] | None = None,
        **invocation_options: Any,
    ) -> BuilderResult:
        """Run the builder as a replica would.

        Args:
            deployment: The model's deployment; a healthy one by default.
            progress: Collects every progress update sent, with no wait between
                them; none is collected when omitted.
            **invocation_options: Overrides of the invocation's fields.

        Returns:
            The builder's result.
        """
        module_name, _, class_name = self.spec.implementation.partition(":")
        implementation = getattr(importlib.import_module(module_name), class_name)
        running = deployment or healthy_deployment(self.model)
        storage = SharedStorageAccessFactory([self.root])
        application = BuilderApplication(
            self.spec,
            implementation(ReplicaSettings(num_threads=1, max_concurrent_tasks=2)),
            {
                "shared_storage": storage
                if progress is None
                else _RecordingProgress(storage, progress)
            },
            model_handles=lambda _model_id: running,  # type: ignore[arg-type, return-value]
            progress_interval_seconds=0,
        )
        options = {
            "inputs": self.inputs(),
            "configuration": self.configuration or None,
            **invocation_options,
        }
        return await application.run(
            invocation("shared_storage", {"root": str(self.root)}, **options),
            self.model,
        )

    def leftovers(self) -> list[Path]:
        """Files besides the inputs, e.g. temporary or partial artifacts."""
        inputs = {self.root / "slide.tiff"}
        return [p for p in self.root.rglob("*") if p.is_file() and p not in inputs]

    async def test_progress_rises_while_the_builder_works(self) -> None:
        sent: list[float] = []

        await self.run_builder(progress=sent)

        self.assertTrue(sent, "no progress was reported")
        self.assertEqual(sent, sorted(sent))
        self.assertTrue(all(0 < fraction <= 1 for fraction in sent), sent)

    async def test_outputs_are_the_declared_ones_at_their_default_paths(self) -> None:
        result = await self.run_builder()

        self.assertEqual(result.outputs.keys(), self.spec.outputs.keys())
        for name, output in result.outputs.items():
            declared = self.spec.outputs[name]
            self.assertIs(output.kind, declared.kind)
            if isinstance(output, SharedArtifact):
                self.assertEqual(output.media_type, declared.media_type)
                self.assertEqual(
                    Path(output.path), self.root.resolve() / str(declared.default_path)
                )
        self.assertEqual(
            sorted(self.leftovers()),
            sorted(
                Path(o.path)
                for o in result.outputs.values()
                if isinstance(o, SharedArtifact)
            ),
        )

    async def test_an_output_path_places_the_artifact(self) -> None:
        if sum(o.default_path is not None for o in self.spec.outputs.values()) != 1:
            self.skipTest("output_path needs exactly one artifact output")

        result = await self.run_builder(output_path="out/result")

        (artifact,) = [
            o for o in result.outputs.values() if isinstance(o, SharedArtifact)
        ]
        self.assertEqual(Path(artifact.path), self.root.resolve() / "out/result")

    async def test_an_unreadable_slide_is_an_invalid_input(self) -> None:
        (self.root / "slide.tiff").write_text("not a slide")

        with self.assertRaises(InvalidInvocation):
            await self.run_builder()
        self.assertEqual(self.leftovers(), [])

    async def test_a_model_running_unlike_its_declaration_is_never_called(
        self,
    ) -> None:
        deployment = healthy_deployment(self.model)
        deployment.get_config.function = lambda: (
            running_config(self.model) | {"tile_size": self.model.tile_size * 2}
        )

        with self.assertRaises(BuilderFailure) as raised:
            await self.run_builder(deployment)

        self.assertEqual(raised.exception.code, FailureCode.MODEL_CONTRACT_MISMATCH)
        calls = [
            method.calls
            for name, method in vars(deployment).items()
            if name != "get_config"
        ]
        self.assertEqual(calls, [0])
        self.assertEqual(self.leftovers(), [])

    async def test_a_failing_model_leaves_no_artifact(self) -> None:
        def fail(_tile: object) -> object:
            raise RuntimeError("CUDA out of memory")

        deployment = healthy_deployment(self.model)
        for name, method in vars(deployment).items():
            if name != "get_config":
                method.function = fail

        with self.assertRaises(BuilderFailure) as raised:
            await self.run_builder(deployment)

        self.assertEqual(raised.exception.code, FailureCode.MODEL_FAILED)
        self.assertTrue(raised.exception.retryable)
        self.assertEqual(self.leftovers(), [])


class _RecordingProgress(InputAccessFactory):
    """Shared storage access whose progress updates are collected."""

    def __init__(self, factory: SharedStorageAccessFactory, sent: list[float]) -> None:
        self._factory = factory
        self._sent = sent

    def create(
        self,
        invocation: Invocation,
        execution_context: ExecutionContext,
        model: ModelBinding,
    ) -> InputAccess:
        return _RecordingAccess(
            self._factory.create(invocation, execution_context, model), self._sent
        )


class _RecordingAccess(InputAccess):
    """Another access, with the progress it is sent collected."""

    def __init__(self, access: InputAccess, sent: list[float]) -> None:
        self._access = access
        self._sent = sent

    def input_path(self, name: str) -> Path:
        return self._access.input_path(name)

    def output_path(self, relative_path: str) -> Path:
        return self._access.output_path(relative_path)

    def is_visible_artifact(self, path: Path) -> bool:
        return self._access.is_visible_artifact(path)

    async def report_progress(self, fraction: float) -> None:
        self._sent.append(fraction)

    async def aclose(self) -> None:
        await self._access.aclose()
