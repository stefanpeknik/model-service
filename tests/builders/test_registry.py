"""Check builder declarations, configuration resolution and invocation validation."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from model_service.builders.contracts import (
    InputAccessContext,
    InvalidInvocation,
    Invocation,
    SharedPathInput,
    UnknownBuilderError,
)
from model_service.builders.registry import BUILDERS, get_builder
from model_service.builders.spec import InputSpec
from tests.support.builder_fixtures import StrideConfiguration
from tests.support.builder_fixtures import spec as test_spec


SPEC = test_spec(
    inputs={"slide": InputSpec(), "region_of_interest": InputSpec(required=False)},
    configuration=StrideConfiguration,
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
    def test_unknown_builder_is_rejected(self) -> None:
        with self.assertRaises(UnknownBuilderError):
            get_builder("missing")

    def test_invalid_configuration_is_an_invalid_invocation(self) -> None:
        for overrides in (
            {"stride_fraction": "dense"},
            {"encoding": "uint8"},
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaises(InvalidInvocation),
            ):
                SPEC.resolve_configuration(overrides)

    def test_validate_invocation_checks_inputs_deadline_and_configuration(
        self,
    ) -> None:
        SPEC.check_invocation(invocation())
        SPEC.check_invocation(invocation(inputs={"slide": SharedPathInput("s")}))
        cases: dict[str, Invocation] = {
            "missing inputs": invocation(inputs={}),
            "unknown inputs": invocation(
                inputs={"slide": SharedPathInput("s"), "extra": SharedPathInput("e")}
            ),
            "timezone": invocation(deadline_at=datetime.now() + timedelta(minutes=1)),
            "future": invocation(deadline_at=datetime.now(UTC) - timedelta(seconds=1)),
            "configuration is invalid": invocation(
                resolved_configuration={"stride_fraction": "dense"}
            ),
        }
        for message, value in cases.items():
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(InvalidInvocation, message),
            ):
                SPEC.check_invocation(value)


class BuilderWiringTests(unittest.TestCase):
    """A registered builder's pieces must agree; a new builder is checked here.

    `tests/builders/test_deployment_files.py` checks the Helm side.
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
