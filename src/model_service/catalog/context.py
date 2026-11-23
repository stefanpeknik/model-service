"""What a document kind resolves one document with.

The context resolves `{use: <name>, ...options}` values against the component
registry, validates plain fields, follows references to earlier kinds and
collects every mistake in the standard format, attributed to the file that
supplied the value (the base file or the environment's patch).
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

from pydantic import ValidationError

from model_service.catalog.errors import CatalogError, render_path
from model_service.catalog.validation import problems
from model_service.components.kinds import KINDS
from model_service.components.problems import Problem


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pydantic import BaseModel

    from model_service.catalog.catalog import Entry
    from model_service.catalog.errors import YamlPath
    from model_service.catalog.kinds import DocumentKind
    from model_service.components.component import Component
    from model_service.components.registry import Registry
    from model_service.components.slots import Slot


class References:
    """The documents resolved so far, for references between kinds."""

    def __init__(self, kinds: Mapping[str, DocumentKind]) -> None:
        """Start empty.

        Args:
            kinds: Every document kind by folder.
        """
        self.kinds = kinds
        self.entries: dict[str, dict[str, Entry]] = {f: {} for f in kinds}
        self.failed: dict[str, set[str]] = {f: set() for f in kinds}


class ResolveContext:
    """Resolves the parts of one document and collects its errors."""

    def __init__(
        self,
        registry: Registry,
        references: References,
        file_of: Callable[[YamlPath], str],
        document_id: str = "",
    ) -> None:
        """Bind the registry, earlier documents and file attribution.

        Args:
            registry: The components catalog files may name.
            references: The documents of earlier kinds.
            file_of: The file that supplied the value at a path.
            document_id: The document's ID, its file name without `.yaml`.
        """
        self.document_id = document_id
        self.registry = registry
        self.references = references
        self.file_of = file_of
        self.errors: list[CatalogError] = []

    def error(self, path: YamlPath, what: str, fits: str, hint: str) -> None:
        """Record one mistake at `path`.

        Args:
            path: Where in the document.
            what: What's wrong.
            fits: What would fit instead.
            hint: How to fix it.
        """
        self.errors.append(
            CatalogError(self.file_of(path), render_path(path), what, fits, hint)
        )

    def component(self, slot: Slot, value: object, path: YamlPath) -> Component | None:
        """Resolve a `{use: <name>, ...options}` value into a component.

        Args:
            slot: Where the value sits.
            value: The YAML value; None when the key is missing.
            path: The value's path in the document.

        Returns:
            The component, or None after recording what's wrong.
        """
        usable = [c.name for c in self.registry.usable_in(slot)]
        if not isinstance(value, dict):
            self._not_a_component(slot, value, path, usable)
            return None
        name = value.get("use")
        if not isinstance(name, str):
            self.error(
                (*path, "use"),
                "missing `use:`",
                f"Usable under {slot}: {_names(usable)}",
                "name the component with `use: <name>`",
            )
            return None
        component = self.registry.lookup(slot, name)
        if component is None:
            self._unknown(slot, name, (*path, "use"), usable)
            return None
        options = {k: v for k, v in value.items() if k != "use"}
        try:
            return component.from_options(options)
        except ValidationError as error:
            owner = f"`{component.name}`"
            for loc, problem, hint in problems(
                error, component.Options, noun="option", owner=owner
            ):
                if problem.code == "missing_option":
                    hint += f" under {slot}"
                problem.detail["slot"] = slot
                self._report((*path, *loc), problem, component.hint(problem) or hint)
            return None

    def fields[M: BaseModel](
        self, model: type[M], value: object, path: YamlPath = ()
    ) -> M | None:
        """Validate plain fields of the document with a pydantic model.

        Args:
            model: The fields' model.
            value: The mapping to validate.
            path: Where the mapping sits in the document.

        Returns:
            The validated model, or None after recording what's wrong.
        """
        try:
            return model.model_validate(value)
        except ValidationError as error:
            for loc, problem, hint in problems(
                error, model, noun="key", owner="this file"
            ):
                self._report((*path, *loc), problem, hint)
            return None

    def reference(self, folder: str, value: object, path: YamlPath) -> Entry | None:
        """Follow a reference to a document of an earlier kind.

        A reference to a document that failed to resolve returns None without
        a second error; that document's own errors already say what's wrong.

        Args:
            folder: The referenced kind's folder.
            value: The referenced ID.
            path: Where the reference sits.

        Returns:
            The referenced entry, or None.
        """
        entries = self.references.entries[folder]
        if isinstance(value, str) and value in entries:
            return entries[value]
        if isinstance(value, str) and value in self.references.failed[folder]:
            return None
        title = self.references.kinds[folder].title
        self.error(
            path,
            f"no {title} `{value}`",
            f"{title.capitalize()}s: {_names(sorted(entries))}",
            f"add catalog/{folder}/{value}.yaml or fix the name",
        )
        return None

    def _report(self, path: YamlPath, problem: Problem, hint: str) -> None:
        """Record a component's failed check as an error.

        Args:
            path: Where in the file the component is.
            problem: The failed check.
            hint: How to fix it.
        """
        self.error(path, problem.message, problem.fits, hint)

    def _not_a_component(
        self, slot: Slot, value: object, path: YamlPath, usable: list[str]
    ) -> None:
        """Record a slot value that names no component: missing, or not a mapping.

        Args:
            slot: The slot.
            value: What the file wrote there.
            path: Where in the file.
            usable: The names usable in the slot, for the example.
        """
        if value is None:
            kind = next(k.kind for k in KINDS if slot in k.slots)
            example = usable[0] if usable else "<name>"
            self.error(
                path,
                f"missing `{slot.key}:`",
                f"A {kind}, e.g. `{slot.key}: {{use: {example}, …}}`",
                f"add `{slot.key}:` with `use:` and its options",
            )
            return
        name = value if isinstance(value, str) else "<name>"
        self.error(
            path,
            f"`{value}` is not a mapping",
            f"`{slot.key}: {{use: <name>, <option>: <value>, …}}`",
            f"write `{slot.key}: {{use: {name}}}`",
        )

    def _unknown(
        self, slot: Slot, name: str, path: YamlPath, usable: list[str]
    ) -> None:
        """Record a component name the slot can't take, with a fix.

        A name of another kind's component says where it belongs; any other name
        gets the closest usable one.

        Args:
            slot: The slot.
            name: The name the file wrote.
            path: Where in the file.
            usable: The names usable in the slot.
        """
        fits = f"Usable under {slot}: {_names(usable)}"
        elsewhere = self.registry.named(name)
        if elsewhere:
            other = elsewhere[0]
            keys = list(dict.fromkeys(f"`{s.key}:`" for s in other.slots))
            problem = Problem(
                "wrong_slot",
                f"`{name}` is a {other.kind}, not usable under {slot}",
                fits,
                {"slot": slot},
            )
            self._report(
                path, problem, other.hint(problem) or f"use `{name}` under {_or(keys)}"
            )
            return
        close = difflib.get_close_matches(name, usable, n=1)
        hint = f"did you mean `{close[0]}`?" if close else "pick one of them"
        self.error(path, f"no component `{name}`", fits, hint)


def _names(names: list[str]) -> str:
    """Names for a message, or a note that there are none.

    Args:
        names: Component names.

    Returns:
        The names, comma-separated, or `none yet`.
    """
    return ", ".join(names) or "none yet"


def _or(items: list[str]) -> str:
    """Items joined as alternatives.

    Args:
        items: At least one item.

    Returns:
        `a`, `a or b`, or `a, b or c`.
    """
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} or {items[-1]}"
