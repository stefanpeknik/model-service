"""Each declared output is published by the publisher of its EMPAIA type."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, patch
from zipfile import BadZipFile

from model_service.builders.contracts import (
    BuilderResult,
    InlineOutput,
    OutputKind,
    SharedArtifact,
)
from model_service.builders.heatmap.score_grid import MEDIA_TYPE
from model_service.builders.registry import get_builder
from model_service.builders.spec import OutputSpec
from model_service.providers.empaia.gateway import EmpaiaGatewayError
from model_service.providers.empaia.image_io import _clip
from model_service.providers.empaia.io import BUILDER_IO
from model_service.providers.empaia.provider import EmpaiaProviderAdapter
from model_service.providers.empaia.publishers import (
    PrimitivePublisher,
    PublisherRegistry,
    default_publishers,
)
from model_service.scheduler.providers import (
    ProviderFinishFailureError,
    ProviderSuccess,
)
from tests.support.builder_fixtures import outputs_model


PUBLISHERS = default_publishers()
HEATMAP = get_builder("heatmap").outputs["heatmap"]


class PublisherTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_primitive_is_posted_as_its_declared_type(self) -> None:
        gateway = AsyncMock()

        commit = await PrimitivePublisher("float").prepare(
            gateway, "score", InlineOutput(1)
        )
        gateway.post_primitive.assert_not_awaited()
        await commit()

        gateway.post_primitive.assert_awaited_once_with("score", "float", 1)

    async def test_a_value_of_another_type_is_rejected_before_any_write(self) -> None:
        for output_type, value in (
            ("integer", True),
            ("integer", 0.5),
            ("float", False),
            ("bool", 1),
            ("string", ""),
        ):
            with self.subTest(output_type=output_type, value=value):
                gateway = AsyncMock()
                with self.assertRaises(EmpaiaGatewayError) as raised:
                    await PrimitivePublisher(output_type).prepare(
                        gateway, "score", InlineOutput(value)
                    )
                self.assertFalse(raised.exception.retryable)
                gateway.post_primitive.assert_not_awaited()

    def test_an_output_unlike_its_declaration_is_rejected(self) -> None:
        for output in (
            InlineOutput(0.5),
            SharedArtifact(path="/shared/heatmap.h5", media_type="application/x-hdf5"),
        ):
            with (
                self.subTest(output=output),
                self.assertRaises(EmpaiaGatewayError),
            ):
                PUBLISHERS.publisher_for(
                    "heatmap", "continuous_pixelmap", HEATMAP, output
                )

    def test_an_output_without_a_known_empaia_type_has_no_publisher(self) -> None:
        inline = OutputSpec(kind=OutputKind.INLINE)

        self.assertIsNone(PUBLISHERS.resolve(None, HEATMAP))
        self.assertIsNone(PUBLISHERS.resolve("polygon", inline))
        self.assertIsNotNone(PUBLISHERS.resolve("continuous_pixelmap", HEATMAP))


class RectangleTests(unittest.TestCase):
    def test_a_rectangle_is_clipped_to_the_slide(self) -> None:
        self.assertEqual(_clip(10, 20, 100), (10, 20))
        self.assertEqual(_clip(-100, 200, 1000), (0, 100))
        self.assertEqual(_clip(90, 20, 100), (90, 10))
        # Nothing on the slide: not positive, so the writer refuses it.
        self.assertLessEqual(_clip(-500, 100, 1000)[1], 0)


class FinishTests(unittest.IsolatedAsyncioTestCase):
    async def test_an_output_the_builder_does_not_declare_is_never_published(
        self,
    ) -> None:
        gateway = AsyncMock()
        gateway.set_before_mutation = lambda _callback: None
        provider = EmpaiaProviderAdapter(
            lambda _context: gateway,  # type: ignore[arg-type, return-value]
            allowed_api_urls=("https://empaia.invalid",),
        )

        with self.assertRaises(ProviderFinishFailureError) as raised:
            await provider.finish(
                {"api_url": "https://empaia.invalid", "job_id": "j", "credential": "c"},
                ProviderSuccess(
                    BuilderResult(
                        outputs={
                            "extra": SharedArtifact(
                                path="/shared/x.npz", media_type=MEDIA_TYPE
                            )
                        }
                    ),
                    "heatmap",
                ),
            )

        self.assertFalse(raised.exception.retryable)
        self.assertFalse(raised.exception.outcome_unknown)
        gateway.finalize.assert_not_awaited()

    async def test_an_unreadable_artifact_is_invalid_not_retried(self) -> None:
        pixelmaps = AsyncMock()
        pixelmaps.prepare.side_effect = BadZipFile("truncated")
        gateway = AsyncMock()
        gateway.set_before_mutation = lambda _callback: None
        provider = EmpaiaProviderAdapter(
            lambda _context: gateway,  # type: ignore[arg-type, return-value]
            allowed_api_urls=("https://empaia.invalid",),
            publishers=PublisherRegistry(
                {("continuous_pixelmap", MEDIA_TYPE): pixelmaps}
            ),
        )

        with self.assertRaises(ProviderFinishFailureError) as raised:
            await provider.finish(
                {"api_url": "https://empaia.invalid", "job_id": "j", "credential": "c"},
                ProviderSuccess(
                    BuilderResult(
                        outputs={
                            "heatmap": SharedArtifact(
                                path="/shared/heatmap.npz", media_type=MEDIA_TYPE
                            )
                        }
                    ),
                    "heatmap",
                ),
            )

        self.assertFalse(raised.exception.retryable)
        self.assertFalse(raised.exception.outcome_unknown)

    async def test_an_invalid_output_stops_the_job_before_any_write(self) -> None:
        heatmap = get_builder("heatmap")
        with_score = replace(
            heatmap,
            output_model=outputs_model(
                {**heatmap.outputs, "score": OutputSpec(kind=OutputKind.INLINE)}
            ),
        )
        io = BUILDER_IO["heatmap"]
        io = replace(io, outputs={**io.outputs, "score": {"type": "float"}})
        pixelmaps = AsyncMock()
        publishers = PublisherRegistry(
            {
                ("continuous_pixelmap", MEDIA_TYPE): pixelmaps,
                ("float", None): PrimitivePublisher("float"),
            }
        )
        gateway = AsyncMock()
        gateway.set_before_mutation = lambda _callback: None
        provider = EmpaiaProviderAdapter(
            lambda _context: gateway,  # type: ignore[arg-type, return-value]
            allowed_api_urls=("https://empaia.invalid",),
            publishers=publishers,
            io={"heatmap": io},
        )

        with (
            patch(
                "model_service.providers.empaia.provider.get_builder",
                return_value=with_score,
            ),
            patch(
                "model_service.builders.registry.get_builder",
                return_value=with_score,
            ),
            self.assertRaises(ProviderFinishFailureError) as raised,
        ):
            await provider.finish(
                {"api_url": "https://empaia.invalid", "job_id": "j", "credential": "c"},
                ProviderSuccess(
                    BuilderResult(
                        outputs={
                            "heatmap": SharedArtifact(
                                path="/shared/heatmap.npz", media_type=MEDIA_TYPE
                            ),
                            "score": InlineOutput("high"),
                        }
                    ),
                    "heatmap",
                ),
            )

        # The heatmap was prepared, but its write never ran.
        pixelmaps.prepare.assert_awaited_once()
        pixelmaps.prepare.return_value.assert_not_awaited()
        gateway.post_primitive.assert_not_awaited()
        self.assertFalse(raised.exception.outcome_unknown)


if __name__ == "__main__":
    unittest.main()
