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

from model_service.builder_deployment.runner import BuilderApplication
from model_service.builders.base import ReplicaSettings
from model_service.builders.registry import get_builder
from model_service.contracts.failures import (
    BuilderFailure,
    FailureCode,
    InvalidInvocation,
)
from model_service.contracts.invocation import SharedPathInput
from model_service.contracts.model_declarations import InferenceContract
from model_service.contracts.result import SharedArtifact
from model_service.providers.standalone.shared_storage import SharedStorageAccessFactory
from tests.support.builder_fixtures import (
    FakeDeployment,
    invocation,
    running_config,
    write_slide,
)


if TYPE_CHECKING:
    import unittest

    from model_service.contracts.invocation import BuilderInput
    from model_service.contracts.model_declarations import ModelBinding
    from model_service.contracts.result import BuilderResult

    _Base = unittest.IsolatedAsyncioTestCase
else:
    _Base = object


def healthy_deployment(model: ModelBinding) -> FakeDeployment:
    """A deployment running as declared that answers every tile within the contract."""
    if model.contract is not InferenceContract.SPATIAL_SCORES:
        raise NotImplementedError(f"no healthy deployment for {model.contract}")
    shape = (model.n_channels, model.output_tile_size, model.output_tile_size)
    return FakeDeployment(
        running_config(model),
        predict=lambda _tile: np.full(shape, 0.5, dtype=np.float32),
    )


class BuilderContractTests(_Base):
    builder_id: ClassVar[str]
    # A model of the builder's contract, at 0.5 mpp so the slide is read at level 0.
    model: ClassVar[ModelBinding]

    async def asyncSetUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        # H&E-like pink, so a background filter sees tissue everywhere.
        pixels = np.empty((64, 96, 3), dtype=np.uint8)
        pixels[...] = (200, 120, 160)
        write_slide(self.root / "slide.tiff", pixels)
        self.spec = get_builder(self.builder_id)

    def inputs(self) -> dict[str, BuilderInput]:
        """The invocation's inputs; override for a builder that needs more than a slide."""
        return {"slide": SharedPathInput("slide.tiff")}

    async def run_builder(
        self,
        deployment: FakeDeployment | None = None,
        **invocation_options: Any,
    ) -> BuilderResult:
        module_name, _, class_name = self.spec.implementation.partition(":")
        implementation = getattr(importlib.import_module(module_name), class_name)
        running = deployment or healthy_deployment(self.model)
        application = BuilderApplication(
            self.spec,
            implementation(ReplicaSettings(num_threads=1, max_concurrent_tasks=2)),
            {"shared_storage": SharedStorageAccessFactory([self.root])},
            model_handles=lambda _model_id: running,  # type: ignore[arg-type, return-value]
        )
        options = {"inputs": self.inputs(), **invocation_options}
        return await application.run(
            invocation("shared_storage", {"root": str(self.root)}, **options),
            self.model,
        )

    def leftovers(self) -> list[Path]:
        """Files besides the inputs, e.g. temporary or partial artifacts."""
        inputs = {self.root / "slide.tiff"}
        return [p for p in self.root.rglob("*") if p.is_file() and p not in inputs]

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
