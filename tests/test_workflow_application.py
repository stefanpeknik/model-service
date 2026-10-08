"""Check workflow validation, safe failures, input access, and atomic artifact publication."""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast

from workflow_execution.application import WorkflowApplication, WorkflowContext
from workflow_execution.catalog import (
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
    WorkflowExecutionContext,
    WorkflowFailure,
    WorkflowInvocation,
    WorkflowResult,
)
from workflow_execution.serialization import (
    invocation_from_dict,
    invocation_to_dict,
    result_from_dict,
    result_to_dict,
)
from workflows.heatmap.prostate import ProstateHeatmapWorkflow


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from workflows.heatmap.processor import HeatmapBuildRequest


class FunctionWorkflow:
    """Adapt an async test function into the workflow run interface."""

    def __init__(
        self,
        function: Callable[[WorkflowContext], Awaitable[WorkflowResult]],
    ) -> None:
        """Retain the async computation function to invoke in the workflow wrapper.

        Args:
            function: Async fixture computation to expose through the workflow
                interface.
        """
        self._function = function

    async def run(self, context: WorkflowContext) -> WorkflowResult:
        """Delegate workflow execution to the configured async test function.

        Args:
            context: Invocation-scoped input/output access, resolved configuration, and
                absolute deadline.

        Returns:
            WorkflowResult: Named inline outputs and/or references to completed artifact
                files.
        """
        return await self._function(context)


class RecordingHeatmapProcessor:
    """Verify workflow file/configuration inputs and write a complete placeholder TIFF."""

    def __init__(self) -> None:
        """Start without a recorded processor destination."""
        self.output_path: Path | None = None

    async def build(self, request: HeatmapBuildRequest) -> None:
        """Check input files and expected stride/TIFF tiles, record the temporary path, and write completed fixture bytes.

        Args:
            request: Slide/mask files, destination, stride fraction, and TIFF tile
                geometry.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
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
        """Fail when the supplied fixture path is not a regular file.

        Args:
            path: File, storage locator, or HTTP endpoint selected by this operation.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        if not path.is_file():
            raise AssertionError(f"{path} is not a regular file")


class FailingHeatmapProcessor(RecordingHeatmapProcessor):
    """Write a partial computation artifact before raising a simulated storage failure."""

    async def build(self, request: HeatmapBuildRequest) -> None:
        """Write fixture bytes through the recording processor and then raise a simulated storage disconnection.

        Args:
            request: Slide/mask files, destination, stride fraction, and TIFF tile
                geometry.

        Returns:
            None: No return value.

        Raises:
            OSError: Input, output, or native storage access fails.
        """
        await super().build(request)
        raise OSError("storage disconnected")


class FixedPathAccess:
    """Expose fixed local fixture paths and containment checks for provider-independent workflow tests."""

    def __init__(self, root: Path) -> None:
        """Retain the directory containing the test slide, mask, and outputs.

        Args:
            root: Existing local directory containing permitted input/output storage.
        """
        self._root = root

    def input_path(self, name: str) -> Path:
        """Map slide and region names to their fixed fixture files.

        Args:
            name: Input/output name or configuration field selected by this operation.

        Returns:
            Path: Existing local file for the named input; invalid or unavailable inputs
                raise an invocation error.
        """
        return {
            "slide": self._root / "slide.svs",
            "region_of_interest": self._root / "region-of-interest.tiff",
        }[name]

    def output_path(self, relative_path: str) -> Path:
        """Join the requested output path to the fixture directory.

        Args:
            relative_path: Output path relative to the permitted root.

        Returns:
            Path: Permitted output destination; allocating the path does not itself
                create or complete the file.
        """
        return self._root / relative_path

    def is_visible_artifact(self, path: Path) -> bool:
        """Require a completed regular file within the resolved fixture directory.

        Args:
            path: File, storage locator, or HTTP endpoint selected by this operation.

        Returns:
            bool: True only for a completed regular file visible within this
                invocation's storage root.
        """
        try:
            path.resolve(strict=True).relative_to(self._root.resolve(strict=True))
        except (OSError, ValueError):
            return False
        return path.is_file()

    async def aclose(self) -> None:
        """Complete without cleanup because fixture access owns no external resource.

        Returns:
            None: No return value.
        """
        return None


class FixedPathAccessAdapter:
    """Construct local fixture access without provider communication."""

    def __init__(self, root: Path) -> None:
        """Retain the local fixture root used when constructing input access.

        Args:
            root: Existing local directory containing permitted input/output storage.
        """
        self._root = root

    def create(
        self,
        _invocation: WorkflowInvocation,
        _execution_context: WorkflowExecutionContext,
    ) -> FixedPathAccess:
        """Construct fixed-path input access for the fixture directory.

        Args:
            _invocation: Unused named inputs, resolved configuration, public access
                state, and absolute workflow deadline; retained to match the interface.
            _execution_context: Unused private attempt identity and provider context
                kept separate from the invocation; retained to match the interface.

        Returns:
            FixedPathAccess: Access to the configured fixture input/output files.
        """
        return FixedPathAccess(self._root)


class FailingAccessAdapter:
    """Inject a private acquisition error to check safe workflow error translation."""

    def create(
        self,
        _invocation: WorkflowInvocation,
        _execution_context: WorkflowExecutionContext,
    ) -> FixedPathAccess:
        """Raise a private acquisition error that public workflow failures must conceal.

        Args:
            _invocation: Unused named inputs, resolved configuration, public access
                state, and absolute workflow deadline; retained to match the interface.
            _execution_context: Unused private attempt identity and provider context
                kept separate from the invocation; retained to match the interface.

        Returns:
            FixedPathAccess: Access to the configured fixture input/output files.

        Raises:
            OSError: Input, output, or native storage access fails.
        """
        raise OSError("provider credential is secret")


class TimingOutAccessAdapter:
    """Inject an acquisition timeout distinct from the workflow deadline."""

    def create(
        self,
        _invocation: WorkflowInvocation,
        _execution_context: WorkflowExecutionContext,
    ) -> FixedPathAccess:
        """Raise a provider acquisition timeout before computation begins.

        Args:
            _invocation: Unused named inputs, resolved configuration, public access
                state, and absolute workflow deadline; retained to match the interface.
            _execution_context: Unused private attempt identity and provider context
                kept separate from the invocation; retained to match the interface.

        Returns:
            FixedPathAccess: Access to the configured fixture input/output files.

        Raises:
            TimeoutError: The described request, cleanup, or polling wait exceeds its
                time budget.
        """
        raise TimeoutError("provider request timed out")


class FailingCloseAccess(FixedPathAccess):
    """Inject a private cleanup failure after workflow execution."""

    async def aclose(self) -> None:
        """Raise a private cleanup error that must retain safe failure classification.

        Returns:
            None: No return value.

        Raises:
            OSError: Input, output, or native storage access fails.
        """
        raise OSError("provider credential is secret")


class FailingCloseAccessAdapter(FixedPathAccessAdapter):
    """Construct fixture access whose cleanup raises a private error."""

    def create(
        self,
        _invocation: WorkflowInvocation,
        _execution_context: WorkflowExecutionContext,
    ) -> FailingCloseAccess:
        """Return fixture access configured to fail during cleanup.

        Args:
            _invocation: Unused named inputs, resolved configuration, public access
                state, and absolute workflow deadline; retained to match the interface.
            _execution_context: Unused private attempt identity and provider context
                kept separate from the invocation; retained to match the interface.

        Returns:
            FailingCloseAccess: Fixture access that raises on cleanup.
        """
        return FailingCloseAccess(self._root)


class TimingOutCloseAccess(FixedPathAccess):
    """Inject a resource cleanup timeout distinct from a workflow deadline."""

    async def aclose(self) -> None:
        """Raise a provider cleanup timeout after computation.

        Returns:
            None: No return value.

        Raises:
            TimeoutError: The described request, cleanup, or polling wait exceeds its
                time budget.
        """
        raise TimeoutError("provider cleanup timed out")


class TimingOutCloseAccessAdapter(FixedPathAccessAdapter):
    """Construct fixture access whose close operation times out."""

    def create(
        self,
        _invocation: WorkflowInvocation,
        _execution_context: WorkflowExecutionContext,
    ) -> TimingOutCloseAccess:
        """Return fixture access configured to time out during cleanup.

        Args:
            _invocation: Unused named inputs, resolved configuration, public access
                state, and absolute workflow deadline; retained to match the interface.
            _execution_context: Unused private attempt identity and provider context
                kept separate from the invocation; retained to match the interface.

        Returns:
            TimingOutCloseAccess: Fixture access that raises a timeout on cleanup.
        """
        return TimingOutCloseAccess(self._root)


class WorkflowApplicationTests(unittest.IsolatedAsyncioTestCase):
    """Exercise workflow validation, safe failures, input access, and atomic artifact publication."""

    async def test_prostate_workflow_returns_only_completed_shared_artifact(
        self,
    ) -> None:
        """Verify prostate workflow returns only completed shared artifact.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
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
        """Verify same workflow uses provider input access.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
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
        """Verify partial artifact is removed after processing failure.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
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
        """Verify inline strings remain inline outputs.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        definition = WorkflowDefinition(
            identifier="metadata",
            inputs={},
            outputs={"label": WorkflowValueDefinition(kind="inline")},
        )

        async def run(_context: WorkflowContext) -> WorkflowResult:
            """Execute the scenario workflow fixture to expose the result or failure checked by its enclosing test.

            Args:
                _context: Unused workflow access/configuration or private provider job
                    context, according to the declared type; retained to match the
                    interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.
            """
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
        """Verify invalid input is classified as permanent.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
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
        """Verify invalid output is a deployment contract violation.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        definition = WorkflowDefinition(
            identifier="metadata",
            inputs={},
            outputs={"label": WorkflowValueDefinition(kind="inline")},
        )

        async def run(_context: WorkflowContext) -> WorkflowResult:
            """Execute the scenario workflow fixture to expose the result or failure checked by its enclosing test.

            Args:
                _context: Unused workflow access/configuration or private provider job
                    context, according to the declared type; retained to match the
                    interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.
            """
            return WorkflowResult(outputs={"wrong": InlineOutput(value=1)})

        application = WorkflowApplication(definition, FunctionWorkflow(run))

        with self.assertRaisesRegex(
            WorkflowContractViolation,
            "outputs do not match workflow definition",
        ):
            await application.run(self._empty_invocation())

    async def test_malformed_output_is_a_deployment_contract_violation(self) -> None:
        """Verify malformed output is a deployment contract violation.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        definition = WorkflowDefinition(
            identifier="metadata",
            inputs={},
            outputs={"label": WorkflowValueDefinition(kind="inline")},
        )

        async def run(_context: WorkflowContext) -> WorkflowResult:
            """Execute the scenario workflow fixture to expose the result or failure checked by its enclosing test.

            Args:
                _context: Unused workflow access/configuration or private provider job
                    context, according to the declared type; retained to match the
                    interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.
            """
            return WorkflowResult(
                outputs={"label": cast("Any", object())},
            )

        application = WorkflowApplication(definition, FunctionWorkflow(run))

        with self.assertRaises(WorkflowContractViolation):
            await application.run(self._empty_invocation())

    async def test_non_json_inline_output_is_a_contract_violation(self) -> None:
        """Verify non JSON inline output is a contract violation.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        definition = WorkflowDefinition(
            identifier="metadata",
            inputs={},
            outputs={"label": WorkflowValueDefinition(kind="inline")},
        )

        async def run(_context: WorkflowContext) -> WorkflowResult:
            """Execute the scenario workflow fixture to expose the result or failure checked by its enclosing test.

            Args:
                _context: Unused workflow access/configuration or private provider job
                    context, according to the declared type; retained to match the
                    interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.
            """
            return WorkflowResult(
                outputs={"label": InlineOutput(value=cast("Any", object()))},
            )

        application = WorkflowApplication(definition, FunctionWorkflow(run))

        with self.assertRaises(WorkflowContractViolation):
            await application.run(self._empty_invocation())

    async def test_retryable_workflow_failure_keeps_its_safe_classification(
        self,
    ) -> None:
        """Verify retryable workflow failure keeps its safe classification.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        definition = WorkflowDefinition(identifier="failure", inputs={}, outputs={})

        async def run(_context: WorkflowContext) -> WorkflowResult:
            """Execute the scenario workflow fixture to expose the result or failure checked by its enclosing test.

            Args:
                _context: Unused workflow access/configuration or private provider job
                    context, according to the declared type; retained to match the
                    interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.

            Raises:
                WorkflowFailure: Execution or resource cleanup fails with a safe
                    classified workflow error.
            """
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
        """Verify unexpected failure does not expose raw exception.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        definition = WorkflowDefinition(identifier="failure", inputs={}, outputs={})

        async def run(_context: WorkflowContext) -> WorkflowResult:
            """Execute the scenario workflow fixture to expose the result or failure checked by its enclosing test.

            Args:
                _context: Unused workflow access/configuration or private provider job
                    context, according to the declared type; retained to match the
                    interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.

            Raises:
                RuntimeError: The required lifecycle state or operation result is
                    inconsistent.
            """
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
        """Verify input access failures use the safe workflow classification.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        definition = WorkflowDefinition(identifier="failure", inputs={}, outputs={})

        async def run(_context: WorkflowContext) -> WorkflowResult:
            """Execute the scenario workflow fixture to expose the result or failure checked by its enclosing test.

            Args:
                _context: Unused workflow access/configuration or private provider job
                    context, according to the declared type; retained to match the
                    interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.
            """
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
        """Verify input access timeout is not a workflow deadline.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        definition = WorkflowDefinition(identifier="failure", inputs={}, outputs={})

        async def run(_context: WorkflowContext) -> WorkflowResult:
            """Execute the scenario workflow fixture to expose the result or failure checked by its enclosing test.

            Args:
                _context: Unused workflow access/configuration or private provider job
                    context, according to the declared type; retained to match the
                    interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.
            """
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
        """Verify input access cleanup failures are classified.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        with TemporaryDirectory() as directory:
            definition = WorkflowDefinition(
                identifier="failure",
                inputs={},
                outputs={},
            )

            async def run(_context: WorkflowContext) -> WorkflowResult:
                """Execute the scenario workflow fixture to expose the result or failure checked by its enclosing test.

                Args:
                    _context: Unused workflow access/configuration or private provider
                        job context, according to the declared type; retained to match
                        the interface.

                Returns:
                    WorkflowResult: Named inline outputs and/or references to completed
                        artifact files.
                """
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
        """Verify input access cleanup timeout is not a workflow deadline.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        with TemporaryDirectory() as directory:
            definition = WorkflowDefinition(
                identifier="failure",
                inputs={},
                outputs={},
            )

            async def run(_context: WorkflowContext) -> WorkflowResult:
                """Execute the scenario workflow fixture to expose the result or failure checked by its enclosing test.

                Args:
                    _context: Unused workflow access/configuration or private provider
                        job context, according to the declared type; retained to match
                        the interface.

                Returns:
                    WorkflowResult: Named inline outputs and/or references to completed
                        artifact files.
                """
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
        """Verify absolute deadline cancels workflow.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        definition = WorkflowDefinition(identifier="slow", inputs={}, outputs={})

        async def run(_context: WorkflowContext) -> WorkflowResult:
            """Execute the scenario workflow fixture to expose the result or failure checked by its enclosing test.

            Args:
                _context: Unused workflow access/configuration or private provider job
                    context, according to the declared type; retained to match the
                    interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.
            """
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

    async def test_shared_artifact_must_be_visible_through_input_access(self) -> None:
        """Verify shared artifact must be visible through input access.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
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
                """Execute the scenario workflow fixture to expose the result or failure checked by its enclosing test.

                Args:
                    _context: Unused workflow access/configuration or private provider
                        job context, according to the declared type; retained to match
                        the interface.

                Returns:
                    WorkflowResult: Named inline outputs and/or references to completed
                        artifact files.
                """
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
        """Verify result can mix inline and shared artifact outputs.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
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
                """Execute the scenario workflow fixture to expose the result or failure checked by its enclosing test.

                Args:
                    _context: Unused workflow access/configuration or private provider
                        job context, according to the declared type; retained to match
                        the interface.

                Returns:
                    WorkflowResult: Named inline outputs and/or references to completed
                        artifact files.
                """
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
        """Verify serialized inputs and outputs keep explicit tags.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
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
        """Build a complete shared-storage prostate invocation over local slide/mask fixtures.

        Args:
            root: Existing local directory containing permitted input/output storage.

        Returns:
            WorkflowInvocation: Typed inputs, resolved configuration, access settings,
                and absolute deadline.
        """
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
        """Build an empty workflow invocation rooted in an existing fixture directory.

        Args:
            root: Existing local directory containing permitted input/output storage.

        Returns:
            WorkflowInvocation: Typed inputs, resolved configuration, access settings,
                and absolute deadline.
        """
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
