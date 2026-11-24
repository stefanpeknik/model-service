"""The folder scan finds components by slot and name and refuses ambiguity."""

from __future__ import annotations

import pytest

from components import Batch, Step, slots
from model_service.registry import Registry, RegistryError
from tests.support.components.providers.empaia.converters.text import Text as TextIn
from tests.support.components.providers.empaia.publishers.text import Text as TextOut
from tests.support.components.steps.mean_intensity import MeanIntensity


TEST_COMPONENTS = "tests.support.components"


def test_a_scanned_component_is_found_in_every_slot_its_kind_allows() -> None:
    registry = Registry.scan(TEST_COMPONENTS)

    assert registry.lookup(slots.STEPS, "mean_intensity") is MeanIntensity
    assert registry.lookup(slots.POST, "mean_intensity") is MeanIntensity
    assert registry.lookup(slots.RUNTIME, "mean_intensity") is None


def test_names_are_unique_per_kind_not_across_kinds() -> None:
    registry = Registry.scan(TEST_COMPONENTS)

    assert registry.lookup(slots.EMPAIA_INPUTS, "text") is TextIn
    assert registry.lookup(slots.EMPAIA_OUTPUTS, "text") is TextOut


def test_two_components_of_one_kind_with_one_name_fail_the_scan() -> None:
    with pytest.raises(RegistryError, match=r"step 'twin'.*first.*second"):
        Registry.scan("tests.support.broken_components.duplicate")


def test_a_component_in_another_kinds_folder_fails_the_scan() -> None:
    with pytest.raises(
        RegistryError, match=r"stray_sink is a sink.*misplaced/steps/.*misplaced/sinks/"
    ):
        Registry.scan("tests.support.broken_components.misplaced")


def test_a_name_that_is_not_snake_case_fails_when_the_class_is_created() -> None:
    with pytest.raises(TypeError, match="snake_case"):

        class _Bad(Step):
            name = "Mean-Pool"

            def __call__(self, batch: Batch) -> Batch:
                return batch
