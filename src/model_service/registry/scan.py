"""The component registry: a scan of the folders of `components/`.

Every module in a component folder is imported, and each concrete component
class defined in it is registered under its kind and name. Nothing lists
components by hand, and catalog files name them only by `use: <name>`.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from functools import cache
from typing import TYPE_CHECKING

from components.base import KINDS
from components.base.component import Component


if TYPE_CHECKING:
    from collections.abc import Iterator

    from components.base.slots import Slot


DEFAULT_PACKAGE = "components"


class RegistryError(TypeError):
    """The component folders are inconsistent: a duplicate or misplaced component."""


class Registry:
    """Components by kind and name, as found by a folder scan."""

    def __init__(self, components: dict[tuple[str, str], type[Component]]) -> None:
        """Hold scanned components.

        Args:
            components: Component classes keyed by (kind, name).
        """
        self._components = components

    @classmethod
    def scan(cls, package: str = DEFAULT_PACKAGE) -> Registry:
        """Import every module in the component folders of `package`.

        Args:
            package: The root package holding each kind's `folder`;
                folders it lacks are skipped.

        Returns:
            The registry of every concrete component found.

        Raises:
            RegistryError: If two components of one kind share a name, or a
                component sits in a folder its kind does not belong to.
        """
        found: dict[tuple[str, str], type[Component]] = {}
        errors: list[str] = []
        for kind in KINDS:
            module = f"{package}.{kind.folder.replace('/', '.')}"
            for component in _components_in(module):
                if not issubclass(component, kind):
                    errors.append(
                        f"{component.name} is a {component.kind} and cannot live "
                        f"in {_path(package, kind.folder)}; move "
                        f"{component.__module__} to "
                        f"{_path(package, component.folder)}"
                    )
                    continue
                key = (component.kind, component.name)
                if key in found:
                    errors.append(
                        f"{component.kind} {component.name!r} is defined twice: "
                        f"{found[key].__module__} and {component.__module__}; "
                        "rename one"
                    )
                    continue
                found[key] = component
        if errors:
            raise RegistryError("\n".join(errors))
        return cls(found)

    def lookup(self, slot: Slot, name: str) -> type[Component] | None:
        """The component of `name` that may be used in `slot`.

        Args:
            slot: Where the catalog file names the component.
            name: The `use:` value.

        Returns:
            The component class, or None if no component of that name fits.
        """
        return next((c for c in self.usable_in(slot) if c.name == name), None)

    def named(self, name: str) -> list[type[Component]]:
        """Every component of `name`, of any kind.

        Args:
            name: A component name.

        Returns:
            The matching classes, one per kind at most.
        """
        return [c for c in self if c.name == name]

    def usable_in(self, slot: Slot) -> list[type[Component]]:
        """Every component that may be used in `slot`, sorted by name.

        Args:
            slot: A slot.

        Returns:
            The component classes whose kind allows the slot.
        """
        return sorted((c for c in self if slot in c.slots), key=lambda c: c.name)

    def __iter__(self) -> Iterator[type[Component]]:
        """Every component, by kind and then name."""
        return iter(self._components[key] for key in sorted(self._components))


def _components_in(package: str) -> Iterator[type[Component]]:
    """Import the modules of `package` and yield the components they define.

    Args:
        package: A dotted package, e.g. `components.steps`.

    Yields:
        Each concrete component defined in its modules, not imported ones.

    Raises:
        ModuleNotFoundError: For a missing module other than the package itself.
    """
    try:
        root = importlib.import_module(package)
    except ModuleNotFoundError as error:
        if error.name is not None and package.startswith(error.name):
            return
        raise
    for info in sorted(pkgutil.iter_modules(root.__path__), key=lambda m: m.name):
        if info.name.startswith("_") or info.ispkg:
            continue
        module = importlib.import_module(f"{package}.{info.name}")
        for _, value in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(value, Component)
                and value.__module__ == module.__name__
                and "name" in value.__dict__
            ):
                yield value


def _path(package: str, folder: str) -> str:
    """A scanned folder as the path a reader would open.

    Args:
        package: The scanned root package, e.g. `components`.
        folder: A kind's folder, e.g. `empaia/converters`.

    Returns:
        E.g. `components/empaia/converters/`.
    """
    return f"{package.replace('.', '/')}/{folder}/"


@cache
def default_registry() -> Registry:
    """The registry of this repo's components, scanned once per process."""
    return Registry.scan(DEFAULT_PACKAGE)
