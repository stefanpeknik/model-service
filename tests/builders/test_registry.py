"""Check builder declarations, configuration resolution and invocation validation."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from model_service.builders.contracts import (
    IncompatibleModelError,
    InputAccessContext,
    InvalidInvocation,
    Invocation,
    SharedPathInput,
    SpatialScoresModel,
    TileEmbeddingModel,
    UnknownBuilderError,
)
from model_service.builders.heatmap.declaration import HeatmapConfiguration
from model_service.builders.registry import BUILDERS, get_builder


SPATIAL = SpatialScoresModel(
    id="classifier", tile_size=512, output_tile_size=1, n_channels=1, mpp=0.5
)
EMBEDDING = TileEmbeddingModel(
    id="encoder", tile_size=224, mpp=0.5, summaries=("tile",)
)


def invocation(**overrides: object) -> Invocation:
    values: dict[str, object] = {
        "inputs": {
            "slide": SharedPathInput("slide.tiff"),
            "region_of_interest": SharedPathInput("mask.tiff"),
        },
        "resolved_configuration": None,
        "input_access": InputAccessContext("shared_storage", {"root": "/data"}),
        "deadline_at": datetime.now(UTC) + timedelta(minutes=1),
    }
    values.update(overrides)
    return Invocation(**values)  # type: ignore[arg-type]


class BuilderRegistryTests(unittest.TestCase):
    def test_builders_accept_one_contract_each(self) -> None:
        self.assertEqual(set(BUILDERS), {"heatmap", "tile-embeddings"})
        self.assertEqual(get_builder("heatmap").accepts, SPATIAL.contract)
        self.assertEqual(get_builder("tile-embeddings").accepts, EMBEDDING.contract)
        with self.assertRaises(UnknownBuilderError):
            get_builder("prostate-heatmap")

    def test_configuration_defaults_and_overrides(self) -> None:
        spec = get_builder("heatmap")
        defaults = spec.resolve_configuration(None)
        self.assertEqual(defaults, HeatmapConfiguration())
        assert isinstance(defaults, HeatmapConfiguration)
        self.assertEqual(defaults.stride_fraction, 0.5)
        self.assertTrue(defaults.skip_background)
        resolved = spec.resolve_configuration({"stride_fraction": 1.0})
        assert isinstance(resolved, HeatmapConfiguration)
        self.assertEqual(resolved.stride_fraction, 1.0)
        self.assertTrue(resolved.skip_background)

    def test_invalid_configuration_is_an_invalid_invocation(self) -> None:
        spec = get_builder("heatmap")
        for overrides in (
            {"stride_fraction": 0},
            {"stride_fraction": 1.5},
            {"skip_background": "maybe"},
            {"encoding": "uint8"},
            {"output_path": "heatmap.npz"},
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaises(InvalidInvocation),
            ):
                spec.resolve_configuration(overrides)

    def test_validate_invocation_checks_inputs_deadline_and_configuration(
        self,
    ) -> None:
        spec = get_builder("heatmap")
        spec.check_invocation(invocation())
        spec.check_invocation(invocation(inputs={"slide": SharedPathInput("s")}))
        cases: dict[str, Invocation] = {
            "missing inputs": invocation(inputs={}),
            "unknown inputs": invocation(
                inputs={"slide": SharedPathInput("s"), "extra": SharedPathInput("e")}
            ),
            "timezone": invocation(deadline_at=datetime.now() + timedelta(minutes=1)),
            "future": invocation(deadline_at=datetime.now(UTC) - timedelta(seconds=1)),
            "configuration is invalid": invocation(
                resolved_configuration={"stride_fraction": 2}
            ),
        }
        for message, value in cases.items():
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(InvalidInvocation, message),
            ):
                spec.check_invocation(value)

    def test_validate_model_rejects_other_contracts(self) -> None:
        get_builder("heatmap").check_model(SPATIAL)
        get_builder("tile-embeddings").check_model(EMBEDDING)
        with self.assertRaises(IncompatibleModelError):
            get_builder("heatmap").check_model(EMBEDDING)
        with self.assertRaises(IncompatibleModelError):
            get_builder("tile-embeddings").check_model(SPATIAL)


class BuilderWiringTests(unittest.TestCase):
    """A registered builder's pieces must agree; a new builder is checked here.

    `tests/facade/test_deployment_files.py` checks the Helm side.
    """

    def test_every_builder_implements_its_declaration(self) -> None:
        import importlib
        from typing import get_args, get_origin

        from model_service.builders.base import Builder

        for spec in BUILDERS.values():
            with self.subTest(spec.id):
                module_name, _, class_name = spec.implementation.partition(":")
                implementation = getattr(
                    importlib.import_module(module_name), class_name
                )
                (base,) = [
                    b for b in implementation.__orig_bases__ if get_origin(b) is Builder
                ]
                client, inputs, configuration, outputs = get_args(base)
                self.assertEqual(client.contract, spec.accepts)
                self.assertIs(inputs, spec.input_model)
                self.assertIs(configuration, spec.configuration)
                self.assertIs(outputs, spec.output_model)


if __name__ == "__main__":
    unittest.main()
