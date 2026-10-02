from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast

from workflow_execution.application import WorkflowApplication, WorkflowContext
from workflow_execution.catalog import (
    MissingWorkflowApplicationsError,
    WorkflowCatalog,
    WorkflowDefinition,
    WorkflowValueDefinition,
)
from workflow_execution.contracts import (
    InlineOutput,
    InputAccessContext,
    InvalidWorkflowInvocation,
    ProviderInput,
    SharedArtifact,
    SharedPathInput,
    WorkflowContractViolation,
    WorkflowFailure,
    WorkflowInvocation,
    WorkflowResult,
)
from workflow_execution.prostate_heatmap import ProstateHeatmapWorkflow
from workflow_execution.serialization import (
    invocation_from_dict,
    invocation_to_dict,
    result_from_dict,
    result_to_dict,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from workflow_execution.heatmap import HeatmapBuildRequest


class FunctionWorkflow:
    def __init__(
        self,
        function: Callable[[WorkflowContext], Awaitable[WorkflowResult]],
    ) -> None:
        self._function = function

    async def run(self, context: WorkflowContext) -> WorkflowResult:
        return await self._function(context)


class RecordingHeatmapProcessor:
    def __init__(self) -> None:
        self.output_path: Path | None = None

    async def build(self, request: HeatmapBuildRequest) -> None:
        self.output_path = request.output_path
        self.assert_regular_file(request.slide_path)
        self.assert_regular_file(request.tissue_mask_path)
        if request.stride_fraction != 0.5:
            raise AssertionError("unexpected stride")
        if (request.output_tile_height, request.output_tile_width) != (256, 256):
            raise AssertionError("unexpected output tile size")
        request.output_path.write_bytes(b"complete tiff")

    @staticmethod
    def assert_regular_file(path: Path) -> None:
        if not path.is_file():
            raise AssertionError(f"{path} is not a regular file")


class FailingHeatmapProcessor(RecordingHeatmapProcessor):
    async def build(self, request: HeatmapBuildRequest) -> None:
        await super().build(request)
        raise OSError("storage disconnected")


class FixedPathAccess:
    def __init__(self, root: Path) -> None:
        self._root = root

    def input_path(self, name: str) -> Path:
        return {
            "slide": self._root / "slide.svs",
            "region_of_interest": self._root / "region-of-interest.tiff",
        }[name]

    def output_path(self, relative_path: str) -> Path:
        return self._root / relative_path

    def is_visible_artifact(self, path: Path) -> bool:
        try:
            path.resolve(strict=True).relative_to(self._root.resolve(strict=True))
        except (OSError, ValueError):
            return False
        return path.is_file()

    async def aclose(self) -> None:
        return None


class FixedPathAccessAdapter:
    def __init__(self, root: Path) -> None:
        self._root = root

    def create(self, _invocation: WorkflowInvocation) -> FixedPathAccess:
        return FixedPathAccess(self._root)


class FailingAccessAdapter:
    def create(self, _invocation: WorkflowInvocation) -> FixedPathAccess:
        raise OSError("provider credential is secret")


class TimingOutAccessAdapter:
    def create(self, _invocation: WorkflowInvocation) -> FixedPathAccess:
        raise TimeoutError("provider request timed out")


class FailingCloseAccess(FixedPathAccess):
    async def aclose(self) -> None:
        raise OSError("provider credential is secret")


class FailingCloseAccessAdapter(FixedPathAccessAdapter):
    def create(self, _invocation: WorkflowInvocation) -> FailingCloseAccess:
        return FailingCloseAccess(self._root)


class TimingOutCloseAccess(FixedPathAccess):
    async def aclose(self) -> None:
        raise TimeoutError("provider cleanup timed out")


class TimingOutCloseAccessAdapter(FixedPathAccessAdapter):
    def create(self, _invocation: WorkflowInvocation) -> TimingOutCloseAccess:
        return TimingOutCloseAccess(self._root)


class WorkflowApplicationTests(unittest.IsolatedAsyncioTestCase):
    async def test_prostate_workflow_returns_only_completed_shared_artifact(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "slide.svs").touch()
            (root / "region-of-interest.tiff").touch()
            processor = RecordingHeatmapProcessor()
            application = WorkflowApplication(
                WorkflowCatalog.default().get("prostate-heatmap"),
                ProstateHeatmapWorkflow(processor),
            )

            result = await application.run(self._prostate_invocation(root))

            artifact = result.outputs["heatmap"]
            self.assertIsInstance(artifact, SharedArtifact)
            assert isinstance(artifact, SharedArtifact)
            self.assertEqual(artifact.path, str(root / "outputs" / "heatmap.tiff"))
            self.assertEqual(artifact.media_type, "image/tiff")
            self.assertEqual(Path(artifact.path).read_bytes(), b"complete tiff")
            self.assertIsNotNone(processor.output_path)
            assert processor.output_path is not None
            self.assertNotEqual(processor.output_path, Path(artifact.path))
            self.assertFalse(processor.output_path.exists())

    async def test_same_workflow_uses_provider_input_access(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "slide.svs").touch()
            (root / "region-of-interest.tiff").touch()
            application = WorkflowApplication(
                WorkflowCatalog.default().get("prostate-heatmap"),
                ProstateHeatmapWorkflow(RecordingHeatmapProcessor()),
                access_adapters={"fake_provider": FixedPathAccessAdapter(root)},
            )
            invocation = self._prostate_invocation(root)
            provider_invocation = WorkflowInvocation(
                inputs={
                    "slide": ProviderInput(name="slide"),
                    "region_of_interest": ProviderInput(name="region_of_interest"),
                },
                resolved_configuration=invocation.resolved_configuration,
                input_access=InputAccessContext(kind="fake_provider", state={}),
                deadline_at=invocation.deadline_at,
            )

            result = await application.run(provider_invocation)

            self.assertIsInstance(result.outputs["heatmap"], SharedArtifact)

    async def test_partial_artifact_is_removed_after_processing_failure(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "slide.svs").touch()
            (root / "region-of-interest.tiff").touch()
            processor = FailingHeatmapProcessor()
            application = WorkflowApplication(
                WorkflowCatalog.default().get("prostate-heatmap"),
                ProstateHeatmapWorkflow(processor),
            )

            with self.assertRaises(WorkflowFailure):
                await application.run(self._prostate_invocation(root))

            self.assertFalse((root / "outputs" / "heatmap.tiff").exists())
            self.assertIsNotNone(processor.output_path)
            assert processor.output_path is not None
            self.assertFalse(processor.output_path.exists())

    async def test_inline_strings_remain_inline_outputs(self) -> None:
        definition = WorkflowDefinition(
            identifier="metadata",
            inputs={},
            outputs={"label": WorkflowValueDefinition(kind="inline")},
        )

        async def run(_context: WorkflowContext) -> WorkflowResult:
            return WorkflowResult(
                outputs={"label": InlineOutput(value="slides/result.tiff")}
            )

        application = WorkflowApplication(definition, FunctionWorkflow(run))

        result = await application.run(
            WorkflowInvocation(
                inputs={},
                resolved_configuration={},
                input_access=InputAccessContext(
                    kind="shared_storage",
                    state={"root": "/tmp"},
                ),
                deadline_at=datetime.now(UTC) + timedelta(minutes=1),
            )
        )

        self.assertEqual(
            result.outputs["label"],
            InlineOutput(value="slides/result.tiff"),
        )

    async def test_invalid_input_is_classified_as_permanent(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "slide.svs").touch()
            application = WorkflowApplication(
                WorkflowCatalog.default().get("prostate-heatmap"),
                ProstateHeatmapWorkflow(RecordingHeatmapProcessor()),
            )
            invocation = self._prostate_invocation(root)
            invocation.inputs.pop("region_of_interest")

            with self.assertRaisesRegex(
                InvalidWorkflowInvocation,
                "inputs do not match workflow definition",
            ):
                await application.run(invocation)

    async def test_invalid_output_is_a_deployment_contract_violation(self) -> None:
        definition = WorkflowDefinition(
            identifier="metadata",
            inputs={},
            outputs={"label": WorkflowValueDefinition(kind="inline")},
        )

        async def run(_context: WorkflowContext) -> WorkflowResult:
            return WorkflowResult(outputs={"wrong": InlineOutput(value=1)})

        application = WorkflowApplication(definition, FunctionWorkflow(run))

        with self.assertRaisesRegex(
            WorkflowContractViolation,
            "outputs do not match workflow definition",
        ):
            await application.run(self._empty_invocation())

    async def test_malformed_output_is_a_deployment_contract_violation(self) -> None:
        definition = WorkflowDefinition(
            identifier="metadata",
            inputs={},
            outputs={"label": WorkflowValueDefinition(kind="inline")},
        )

        async def run(_context: WorkflowContext) -> WorkflowResult:
            return WorkflowResult(
                outputs={"label": cast("Any", object())},
            )

        application = WorkflowApplication(definition, FunctionWorkflow(run))

        with self.assertRaises(WorkflowContractViolation):
            await application.run(self._empty_invocation())

    async def test_non_json_inline_output_is_a_contract_violation(self) -> None:
        definition = WorkflowDefinition(
            identifier="metadata",
            inputs={},
            outputs={"label": WorkflowValueDefinition(kind="inline")},
        )

        async def run(_context: WorkflowContext) -> WorkflowResult:
            return WorkflowResult(
                outputs={"label": InlineOutput(value=cast("Any", object()))},
            )

        application = WorkflowApplication(definition, FunctionWorkflow(run))

        with self.assertRaises(WorkflowContractViolation):
            await application.run(self._empty_invocation())

    async def test_retryable_workflow_failure_keeps_its_safe_classification(
        self,
    ) -> None:
        definition = WorkflowDefinition(identifier="failure", inputs={}, outputs={})

        async def run(_context: WorkflowContext) -> WorkflowResult:
            raise WorkflowFailure(
                code="shared_storage_unavailable",
                safe_message="shared storage is unavailable",
                retryable=True,
            )

        application = WorkflowApplication(definition, FunctionWorkflow(run))

        with self.assertRaises(WorkflowFailure) as raised:
            await application.run(self._empty_invocation())

        self.assertEqual(raised.exception.code, "shared_storage_unavailable")
        self.assertTrue(raised.exception.retryable)

    async def test_unexpected_failure_does_not_expose_raw_exception(self) -> None:
        definition = WorkflowDefinition(identifier="failure", inputs={}, outputs={})

        async def run(_context: WorkflowContext) -> WorkflowResult:
            raise RuntimeError("database password is secret")

        application = WorkflowApplication(definition, FunctionWorkflow(run))

        with self.assertRaises(WorkflowFailure) as raised:
            await application.run(self._empty_invocation())

        self.assertEqual(raised.exception.code, "workflow_internal_error")
        self.assertEqual(
            raised.exception.safe_message,
            "workflow failed unexpectedly",
        )
        self.assertNotIn("secret", str(raised.exception))

    async def test_input_access_failures_use_the_safe_workflow_classification(
        self,
    ) -> None:
        definition = WorkflowDefinition(identifier="failure", inputs={}, outputs={})

        async def run(_context: WorkflowContext) -> WorkflowResult:
            return WorkflowResult(outputs={})

        invocation = self._empty_invocation()
        invocation = WorkflowInvocation(
            inputs={},
            resolved_configuration={},
            input_access=InputAccessContext(kind="failing", state={}),
            deadline_at=invocation.deadline_at,
        )
        application = WorkflowApplication(
            definition,
            FunctionWorkflow(run),
            access_adapters={"failing": FailingAccessAdapter()},
        )

        with self.assertRaises(WorkflowFailure) as raised:
            await application.run(invocation)

        self.assertEqual(raised.exception.code, "workflow_internal_error")
        self.assertNotIn("secret", str(raised.exception))

    async def test_input_access_timeout_is_not_a_workflow_deadline(self) -> None:
        definition = WorkflowDefinition(identifier="failure", inputs={}, outputs={})

        async def run(_context: WorkflowContext) -> WorkflowResult:
            return WorkflowResult(outputs={})

        invocation = self._empty_invocation()
        invocation = WorkflowInvocation(
            inputs={},
            resolved_configuration={},
            input_access=InputAccessContext(kind="timing_out", state={}),
            deadline_at=invocation.deadline_at,
        )
        application = WorkflowApplication(
            definition,
            FunctionWorkflow(run),
            access_adapters={"timing_out": TimingOutAccessAdapter()},
        )

        with self.assertRaises(WorkflowFailure) as raised:
            await application.run(invocation)

        self.assertEqual(raised.exception.code, "workflow_internal_error")
        self.assertTrue(raised.exception.retryable)

    async def test_input_access_cleanup_failures_are_classified(self) -> None:
        with TemporaryDirectory() as directory:
            definition = WorkflowDefinition(
                identifier="failure",
                inputs={},
                outputs={},
            )

            async def run(_context: WorkflowContext) -> WorkflowResult:
                return WorkflowResult(outputs={})

            invocation = self._empty_invocation()
            invocation = WorkflowInvocation(
                inputs={},
                resolved_configuration={},
                input_access=InputAccessContext(kind="failing", state={}),
                deadline_at=invocation.deadline_at,
            )
            application = WorkflowApplication(
                definition,
                FunctionWorkflow(run),
                access_adapters={"failing": FailingCloseAccessAdapter(Path(directory))},
            )

            with self.assertRaises(WorkflowFailure) as raised:
                await application.run(invocation)

            self.assertEqual(raised.exception.code, "workflow_internal_error")
            self.assertNotIn("secret", str(raised.exception))

    async def test_input_access_cleanup_timeout_is_not_a_workflow_deadline(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            definition = WorkflowDefinition(
                identifier="failure",
                inputs={},
                outputs={},
            )

            async def run(_context: WorkflowContext) -> WorkflowResult:
                return WorkflowResult(outputs={})

            invocation = self._empty_invocation()
            invocation = WorkflowInvocation(
                inputs={},
                resolved_configuration={},
                input_access=InputAccessContext(kind="timing_out", state={}),
                deadline_at=invocation.deadline_at,
            )
            application = WorkflowApplication(
                definition,
                FunctionWorkflow(run),
                access_adapters={
                    "timing_out": TimingOutCloseAccessAdapter(Path(directory))
                },
            )

            with self.assertRaises(WorkflowFailure) as raised:
                await application.run(invocation)

            self.assertEqual(raised.exception.code, "workflow_internal_error")
            self.assertTrue(raised.exception.retryable)

    async def test_absolute_deadline_cancels_workflow(self) -> None:
        definition = WorkflowDefinition(identifier="slow", inputs={}, outputs={})

        async def run(_context: WorkflowContext) -> WorkflowResult:
            await asyncio.sleep(60)
            return WorkflowResult(outputs={})

        application = WorkflowApplication(definition, FunctionWorkflow(run))
        invocation = self._empty_invocation()
        invocation = WorkflowInvocation(
            inputs=invocation.inputs,
            resolved_configuration=invocation.resolved_configuration,
            input_access=invocation.input_access,
            deadline_at=datetime.now(UTC) + timedelta(milliseconds=1),
        )

        with self.assertRaises(WorkflowFailure) as raised:
            await asyncio.wait_for(application.run(invocation), timeout=0.1)

        self.assertEqual(raised.exception.code, "workflow_deadline_exceeded")
        self.assertFalse(raised.exception.retryable)

    def test_catalog_reports_declared_workflow_without_application(self) -> None:
        catalog = WorkflowCatalog.default()

        with self.assertRaises(MissingWorkflowApplicationsError) as raised:
            catalog.validate_deployed(set())

        self.assertEqual(raised.exception.workflow_ids, ("prostate-heatmap",))

    def test_repo_catalog_matches_enabled_helm_applications(self) -> None:
        catalog = WorkflowCatalog.default()

        from workflow_execution.deployment_validation import validate_helm_applications

        validate_helm_applications(
            catalog,
            Path("helm/rayservice/values.yaml"),
            Path("helm/rayservice/applications"),
        )

    async def test_shared_artifact_must_be_visible_through_input_access(self) -> None:
        with (
            TemporaryDirectory() as shared_directory,
            TemporaryDirectory() as private_directory,
        ):
            private_artifact = Path(private_directory) / "private.tiff"
            private_artifact.touch()
            definition = WorkflowDefinition(
                identifier="artifact",
                inputs={},
                outputs={
                    "heatmap": WorkflowValueDefinition(
                        kind="shared_artifact",
                        media_type="image/tiff",
                    )
                },
            )

            async def run(_context: WorkflowContext) -> WorkflowResult:
                return WorkflowResult(
                    outputs={
                        "heatmap": SharedArtifact(
                            path=str(private_artifact),
                            media_type="image/tiff",
                        )
                    }
                )

            application = WorkflowApplication(definition, FunctionWorkflow(run))

            with self.assertRaises(WorkflowContractViolation):
                await application.run(self._empty_invocation(Path(shared_directory)))

    async def test_result_can_mix_inline_and_shared_artifact_outputs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "result.tiff"
            artifact.touch()
            definition = WorkflowDefinition(
                identifier="mixed",
                inputs={},
                outputs={
                    "score": WorkflowValueDefinition(kind="inline"),
                    "heatmap": WorkflowValueDefinition(
                        kind="shared_artifact",
                        media_type="image/tiff",
                    ),
                },
            )

            async def run(_context: WorkflowContext) -> WorkflowResult:
                return WorkflowResult(
                    outputs={
                        "score": InlineOutput(value=0.9),
                        "heatmap": SharedArtifact(
                            path=str(artifact),
                            media_type="image/tiff",
                        ),
                    }
                )

            application = WorkflowApplication(definition, FunctionWorkflow(run))

            result = await application.run(self._empty_invocation(root))

            self.assertEqual(result.outputs["score"], InlineOutput(value=0.9))
            self.assertEqual(
                result.outputs["heatmap"],
                SharedArtifact(path=str(artifact), media_type="image/tiff"),
            )

    def test_serialized_inputs_and_outputs_keep_explicit_tags(self) -> None:
        invocation = invocation_from_dict(
            {
                "inputs": {"slide": {"kind": "shared_path", "path": "slide.svs"}},
                "resolved_configuration": {},
                "input_access": {
                    "kind": "shared_storage",
                    "state": {"root": "/shared"},
                },
                "deadline_at": "2026-09-16T12:00:00+00:00",
            }
        )

        self.assertEqual(invocation.inputs["slide"].kind, "shared_path")
        self.assertEqual(
            invocation_from_dict(invocation_to_dict(invocation)), invocation
        )
        serialized_result = {
            "outputs": {
                "label": {
                    "kind": "inline",
                    "value": "outputs/heatmap.tiff",
                }
            }
        }
        self.assertEqual(
            result_to_dict(
                WorkflowResult(
                    outputs={"label": InlineOutput(value="outputs/heatmap.tiff")}
                )
            ),
            serialized_result,
        )
        self.assertEqual(
            result_from_dict(serialized_result),
            WorkflowResult(outputs={"label": InlineOutput("outputs/heatmap.tiff")}),
        )

    @staticmethod
    def _prostate_invocation(root: Path) -> WorkflowInvocation:
        return WorkflowInvocation(
            inputs={
                "slide": SharedPathInput(path="slide.svs"),
                "region_of_interest": SharedPathInput(path="region-of-interest.tiff"),
            },
            resolved_configuration={
                "output_path": "outputs/heatmap.tiff",
                "stride_fraction": 0.5,
                "output_tile_height": 256,
                "output_tile_width": 256,
            },
            input_access=InputAccessContext(
                kind="shared_storage",
                state={"root": str(root)},
            ),
            deadline_at=datetime.now(UTC) + timedelta(minutes=1),
        )

    @staticmethod
    def _empty_invocation(root: Path = Path("/tmp")) -> WorkflowInvocation:
        return WorkflowInvocation(
            inputs={},
            resolved_configuration={},
            input_access=InputAccessContext(
                kind="shared_storage",
                state={"root": str(root)},
            ),
            deadline_at=datetime.now(UTC) + timedelta(minutes=1),
        )


if __name__ == "__main__":
    unittest.main()
