"""Check builder declarations, configuration resolution and invocation validation."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from model_service.builders.registry import BUILDERS, get_builder
from model_service.contracts.failures import (
    InvalidInvocation,
)
from model_service.contracts.invocation import (
    InputAccessContext,
    Invocation,
    SharedPathInput,
)
from model_service.contracts.model_declarations import SpatialScoresModel


SPATIAL = SpatialScoresModel(
    id="classifier", tile_size=512, output_tile_size=1, n_channels=1, mpp=0.5
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


class BuilderWiringTests(unittest.TestCase):
    """A registered builder's pieces must agree; a new builder is checked here.

    `tests/api/test_deployment_files.py` checks the Helm side.
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
