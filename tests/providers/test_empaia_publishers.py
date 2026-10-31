"""Each declared output is published by the publisher of its EMPAIA type."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import AsyncMock

from model_service.builders.contracts import (
    BuilderResult,
    InlineOutput,
    OutputKind,
    SharedArtifact,
    SpatialScoresModel,
)
from model_service.builders.registry import OutputSpec, get_builder
from model_service.builders.score_grid import MEDIA_TYPE
from model_service.providers.empaia.apps import AppMode, EmpaiaListing, app_definitions
from model_service.providers.empaia.gateway import EmpaiaGatewayError
from model_service.providers.empaia.provider import EmpaiaProviderAdapter
from model_service.providers.empaia.publishers import (
    PrimitivePublisher,
    default_publishers,
)
from model_service.scheduler.providers import (
    ProviderFinishFailureError,
    ProviderSuccess,
)


PUBLISHERS = default_publishers()
HEATMAP = get_builder("heatmap").outputs["heatmap"]
LISTING = EmpaiaListing(
    name="Prostate cancer",
    name_short="ProstHM",
    description="Finds things.",
    modes=(AppMode.PREPROCESSING,),
)


def model(n_channels: int) -> SpatialScoresModel:
    return SpatialScoresModel(
        id="prostate-classifier-1",
        tile_size=512,
        output_tile_size=1,
        n_channels=n_channels,
        mpp=0.468,
    )


class PublisherTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_primitive_is_posted_as_its_declared_type(self) -> None:
        gateway = AsyncMock()

        await PrimitivePublisher("float").publish(gateway, "score", InlineOutput(1))

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
                    await PrimitivePublisher(output_type).publish(
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
                PUBLISHERS.publisher_for("heatmap", HEATMAP, output)

    def test_an_output_without_a_known_empaia_type_has_no_publisher(self) -> None:
        embeddings = get_builder("tile-embeddings").outputs["embeddings"]
        polygon = OutputSpec(kind=OutputKind.INLINE, empaia={"type": "polygon"})

        self.assertIsNone(PUBLISHERS.resolve(embeddings))
        self.assertIsNone(PUBLISHERS.resolve(polygon))
        self.assertIsNotNone(PUBLISHERS.resolve(HEATMAP))


class AppPublishabilityTests(unittest.TestCase):
    def test_only_models_their_publishers_take_become_apps(self) -> None:
        single = app_definitions(
            [model(1)], {"prostate-classifier-1": LISTING}, PUBLISHERS
        )
        with self.assertLogs("model_service.providers.empaia.apps", "INFO") as logs:
            multi = app_definitions(
                [model(3)], {"prostate-classifier-1": LISTING}, PUBLISHERS
            )

        self.assertEqual(len(single), 1)
        self.assertEqual(multi, [])
        self.assertIn("nothing publishes heatmap", logs.output[0])

    def test_an_output_nothing_publishes_withholds_the_app(self) -> None:
        heatmap = get_builder("heatmap")
        with_polygons = replace(
            heatmap,
            outputs={
                **heatmap.outputs,
                "glands": OutputSpec(
                    kind=OutputKind.INLINE, empaia={"type": "polygon"}
                ),
            },
        )

        with self.assertLogs("model_service.providers.empaia.apps", "INFO"):
            apps = app_definitions(
                [model(1)],
                {"prostate-classifier-1": LISTING},
                PUBLISHERS,
                [with_polygons],
            )

        self.assertEqual(apps, [])


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


if __name__ == "__main__":
    unittest.main()
