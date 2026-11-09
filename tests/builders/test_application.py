"""Check builder invocation validation, safe failures, outputs and the HTTP seam."""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast

from model_service.builders.application import BuilderApplication
from model_service.builders.artifacts import atomic_artifact
from model_service.builders.contracts import (
    BuilderFailure,
    BuilderResult,
    ContractViolation,
    IncompatibleModelError,
    InlineOutput,
    InvalidInvocation,
    SharedArtifact,
    SharedPathInput,
)
from model_service.builders.spec import InputSpec
from tests.support.builder_fixtures import (
    EMBEDDING_MODEL,
    SPATIAL_MODEL,
    FunctionBuilder,
    artifact_output,
    inline_output,
    invocation,
    spec,
)


if TYPE_CHECKING:
    from model_service.builders.base import BuilderContext


class FixedAccess:
    def __init__(self, root: Path) -> None:
        self._root = root

    def input_path(self, name: str) -> Path:
        return self._root / name

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


class FixedAccessAdapter:
    def __init__(self, root: Path, access: type[FixedAccess] = FixedAccess) -> None:
        self._root = root
        self._access = access

    def create(self, _invocation: object, _model: object) -> FixedAccess:
        return self._access(self._root)


class FailingCloseAccess(FixedAccess):
    async def aclose(self) -> None:
        raise OSError("provider credential is secret")


class TimingOutCloseAccess(FixedAccess):
    async def aclose(self) -> None:
        raise TimeoutError("provider cleanup timed out")


class RaisingAdapter:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def create(self, _invocation: object, _model: object) -> FixedAccess:
        raise self._error


def run(
    function: Any,
) -> FunctionBuilder:
    return FunctionBuilder(function)


class BuilderApplicationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._directory = TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def application(
        self, builder: FunctionBuilder, **spec_options: Any
    ) -> BuilderApplication:
        return BuilderApplication(
            spec(**spec_options),
            builder,
            {"test": FixedAccessAdapter(self.root)},
        )

    async def test_artifact_is_published_atomically_at_default_path(self) -> None:
        temporaries: list[Path] = []

        async def build(context: BuilderContext) -> BuilderResult:
            destination = context.output_path("artifact")
            with atomic_artifact(destination, inputs=()) as temporary:
                temporaries.append(temporary)
                temporary.write_bytes(b"complete")
            return BuilderResult({"artifact": SharedArtifact(str(destination))})

        result = await self.application(
            run(build), outputs={"artifact": artifact_output()}
        ).run(invocation("test"), SPATIAL_MODEL)

        artifact = result.outputs["artifact"]
        assert isinstance(artifact, SharedArtifact)
        self.assertEqual(Path(artifact.path), self.root / "result.bin")
        self.assertEqual(Path(artifact.path).read_bytes(), b"complete")
        self.assertFalse(temporaries[0].exists())

    async def test_output_path_overrides_the_single_artifact_default(self) -> None:
        async def build(context: BuilderContext) -> BuilderResult:
            destination = context.output_path("artifact")
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"x")
            return BuilderResult({"artifact": SharedArtifact(str(destination))})

        result = await self.application(
            run(build), outputs={"artifact": artifact_output()}
        ).run(invocation("test", output_path="custom/out.bin"), SPATIAL_MODEL)

        self.assertEqual(
            result.outputs["artifact"],
            SharedArtifact(str(self.root / "custom" / "out.bin")),
        )

    async def test_output_path_requires_an_artifact_output(self) -> None:
        async def build(_context: BuilderContext) -> BuilderResult:
            return BuilderResult({"label": InlineOutput("x")})

        with self.assertRaisesRegex(InvalidInvocation, "no artifact output"):
            await self.application(run(build), outputs={"label": inline_output()}).run(
                invocation("test", output_path="out.bin"), SPATIAL_MODEL
            )

    async def test_failed_build_leaves_no_partial_artifact(self) -> None:
        temporaries: list[Path] = []

        async def build(context: BuilderContext) -> BuilderResult:
            with atomic_artifact(context.output_path("artifact"), inputs=()) as temp:
                temporaries.append(temp)
                temp.write_bytes(b"partial")
                raise OSError("storage disconnected")

        with self.assertRaises(BuilderFailure):
            await self.application(
                run(build), outputs={"artifact": artifact_output()}
            ).run(invocation("test"), SPATIAL_MODEL)

        self.assertFalse((self.root / "result.bin").exists())
        self.assertFalse(temporaries[0].exists())

    async def test_optional_inputs_are_none_when_not_supplied(self) -> None:
        seen: dict[str, Path | None] = {}

        async def build(context: BuilderContext) -> BuilderResult:
            seen["slide"] = context.inputs.slide  # type: ignore[attr-defined]
            seen["mask"] = context.inputs.region_of_interest  # type: ignore[attr-defined]
            return BuilderResult({})

        builder = self.application(
            run(build),
            inputs={
                "slide": InputSpec(),
                "region_of_interest": InputSpec(required=False),
            },
        )
        await builder.run(
            invocation("test", inputs={"slide": SharedPathInput("slide.tiff")}),
            SPATIAL_MODEL,
        )
        self.assertEqual(seen, {"slide": self.root / "slide", "mask": None})

        await builder.run(
            invocation(
                "test",
                inputs={
                    "slide": SharedPathInput("slide.tiff"),
                    "region_of_interest": SharedPathInput("mask.tiff"),
                },
            ),
            SPATIAL_MODEL,
        )
        self.assertEqual(seen["mask"], self.root / "region_of_interest")

    async def test_incompatible_model_is_rejected_before_access(self) -> None:
        async def build(_context: BuilderContext) -> BuilderResult:
            self.fail("incompatible models must not run")

        application = BuilderApplication(
            spec(),
            run(build),
            {"test": RaisingAdapter(AssertionError("access must not open"))},
        )
        with self.assertRaises(IncompatibleModelError):
            await application.run(invocation("test"), EMBEDDING_MODEL)

    async def test_output_contract_violations(self) -> None:
        artifact = self.root / "result.tiff"
        artifact.touch()
        with TemporaryDirectory() as private:
            invisible = Path(private) / "private.tiff"
            invisible.touch()
            cases: dict[str, tuple[dict[str, Any], BuilderResult]] = {
                "wrong names": (
                    {"label": inline_output()},
                    BuilderResult({"wrong": InlineOutput(1)}),
                ),
                "malformed": (
                    {"label": inline_output()},
                    BuilderResult({"label": cast("Any", object())}),
                ),
                "non-JSON inline": (
                    {"label": inline_output()},
                    BuilderResult({"label": InlineOutput(cast("Any", object()))}),
                ),
                "wrong kind": (
                    {"label": inline_output()},
                    BuilderResult({"label": SharedArtifact(str(artifact))}),
                ),
                "invisible artifact": (
                    {"heatmap": artifact_output(media_type="image/tiff")},
                    BuilderResult(
                        {"heatmap": SharedArtifact(str(invisible), "image/tiff")}
                    ),
                ),
                "wrong media type": (
                    {"heatmap": artifact_output(media_type="image/tiff")},
                    BuilderResult({"heatmap": SharedArtifact(str(artifact), "x/y")}),
                ),
            }
            for name, (outputs, value) in cases.items():

                async def build(
                    _context: BuilderContext, value: BuilderResult = value
                ) -> BuilderResult:
                    return value

                with self.subTest(name), self.assertRaises(ContractViolation):
                    await self.application(run(build), outputs=outputs).run(
                        invocation("test"), SPATIAL_MODEL
                    )

    async def test_builder_failures_keep_their_safe_classification(self) -> None:
        async def build(_context: BuilderContext) -> BuilderResult:
            raise BuilderFailure(
                code="model_unavailable", safe_message="down", retryable=True
            )

        with self.assertRaises(BuilderFailure) as raised:
            await self.application(run(build)).run(invocation("test"), SPATIAL_MODEL)
        self.assertEqual(raised.exception.code, "model_unavailable")
        self.assertTrue(raised.exception.retryable)

    async def test_unexpected_failure_does_not_expose_the_raw_exception(self) -> None:
        async def build(_context: BuilderContext) -> BuilderResult:
            raise RuntimeError("database password is secret")

        with self.assertRaises(BuilderFailure) as raised:
            await self.application(run(build)).run(invocation("test"), SPATIAL_MODEL)
        self.assertEqual(raised.exception.code, "builder_internal_error")
        self.assertEqual(raised.exception.safe_message, "builder failed unexpectedly")
        self.assertTrue(raised.exception.retryable)
        self.assertNotIn("secret", str(raised.exception))

    async def test_access_and_cleanup_failures_are_safe_and_not_deadlines(
        self,
    ) -> None:
        async def build(_context: BuilderContext) -> BuilderResult:
            return BuilderResult({})

        adapters = {
            "acquisition error": RaisingAdapter(OSError("credential is secret")),
            "acquisition timeout": RaisingAdapter(TimeoutError("provider timed out")),
            "cleanup error": FixedAccessAdapter(self.root, FailingCloseAccess),
            "cleanup timeout": FixedAccessAdapter(self.root, TimingOutCloseAccess),
        }
        for name, adapter in adapters.items():
            application = BuilderApplication(spec(), run(build), {"test": adapter})
            with self.subTest(name), self.assertRaises(BuilderFailure) as raised:
                await application.run(invocation("test"), SPATIAL_MODEL)
            self.assertEqual(raised.exception.code, "builder_internal_error")
            self.assertTrue(raised.exception.retryable)
            self.assertNotIn("secret", str(raised.exception))

    async def test_unknown_access_kind_is_invalid(self) -> None:
        async def build(_context: BuilderContext) -> BuilderResult:
            return BuilderResult({})

        with self.assertRaisesRegex(InvalidInvocation, "input access kind"):
            await self.application(run(build)).run(invocation("missing"), SPATIAL_MODEL)

    async def test_absolute_deadline_cancels_the_builder(self) -> None:
        async def build(_context: BuilderContext) -> BuilderResult:
            await asyncio.sleep(60)
            return BuilderResult({})

        request = invocation(
            "test", deadline_at=datetime.now(UTC) + timedelta(milliseconds=20)
        )
        with self.assertRaises(BuilderFailure) as raised:
            await asyncio.wait_for(
                self.application(run(build)).run(request, SPATIAL_MODEL), timeout=1
            )
        self.assertEqual(raised.exception.code, "builder_deadline_exceeded")
        self.assertFalse(raised.exception.retryable)


if __name__ == "__main__":
    unittest.main()
