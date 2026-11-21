"""The base every component kind shares: name, options, columns and hints.

A component is a small class with a `name`, a nested `Options` model and the
method set of its kind. Its module imports only light dependencies (pydantic,
NumPy typing) at top level and heavy ones inside methods, so the API can load
and check every component without Ray, ONNX Runtime, torch or OpenSlide.
"""

from __future__ import annotations

import re
from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar

from model_service.components.options import Options as BaseOptions


if TYPE_CHECKING:
    from model_service.components.columns import Col
    from model_service.components.problems import Problem
    from model_service.components.slots import Slot


NAME = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


class Component(ABC):
    """A reusable piece of code that catalog files name by `use: <name>`.

    Subclasses are kind bases (`Loader`, `Step`, ...) and, below them, concrete
    components that set `name`. A concrete component's declaration is checked
    when its class is created, so a bad one fails at import.

    Attributes:
        kind: The kind's name, set by each kind base.
        slots: Where components of this kind may be named, set by each kind base.
        name: The snake_case name catalog files use.
        Options: The pydantic model of the options a catalog file may give.
        example_options: Valid options used by the conformance suite and as the
            example in generated docs.
        reads: Columns read from each batch.
        adds: Columns added to each batch.
        may_add: Every column the component can add under some options; it
            tells a reader which component would provide a missing column.
        options: The validated options of this instance.
    """

    kind: ClassVar[str]
    slots: ClassVar[tuple[Slot, ...]]
    name: ClassVar[str]
    Options: ClassVar[type[BaseOptions]] = BaseOptions
    example_options: ClassVar[dict[str, Any]] = {}
    reads: ClassVar[dict[str, Col]] = {}
    adds: ClassVar[dict[str, Col]] = {}
    may_add: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Check a concrete component's declaration when its class is created.

        Raises:
            TypeError: If the name is not snake_case or `Options` does not
                derive from the kind's options base.
        """
        super().__init_subclass__(**kwargs)
        if "name" not in cls.__dict__:
            return
        if not isinstance(cls.name, str) or not NAME.match(cls.name):
            raise TypeError(
                f"{cls.__qualname__}.name {cls.name!r} must be snake_case, "
                "e.g. 'cls_mean_pool'"
            )
        base = next(
            b.__dict__.get("Options", BaseOptions)
            for b in cls.__mro__[1:]
            if "kind" in b.__dict__
        )
        if not (isinstance(cls.Options, type) and issubclass(cls.Options, base)):
            raise TypeError(
                f"{cls.__qualname__}.Options must derive from "
                f"{base.__module__}.{base.__qualname__}"
            )
        if "may_add" not in cls.__dict__:
            cls.may_add = frozenset(cls.adds)

    def __init__(self, options: BaseOptions) -> None:
        """Hold validated options.

        Args:
            options: An instance of the component's `Options`.
        """
        self.options = options

    @classmethod
    def from_options(cls, values: dict[str, Any]) -> Component:
        """Validate raw option values and build the component.

        Args:
            values: Option values, e.g. from a catalog file.

        Returns:
            The component.

        Raises:
            pydantic.ValidationError: If the values do not fit `Options`.
        """
        return cls(cls.Options.model_validate(values))

    def columns(self, model: object | None) -> tuple[dict[str, Col], dict[str, Col]]:
        """The columns read and added, with known dimensions filled in.

        Override this when what is read or added depends on options or on the
        model beyond filling in dimensions, e.g. a channel count that follows
        an option.

        Args:
            model: The served-model declaration, or None before it is known.

        Returns:
            The read and added columns.
        """
        return (
            {k: c.resolve(self.options, model) for k, c in self.reads.items()},
            {k: c.resolve(self.options, model) for k, c in self.adds.items()},
        )

    @classmethod
    def hint(cls, problem: Problem) -> str | None:
        """How to fix a problem with this component's use, in one sentence.

        Override it to teach: the generic hint says what failed, a component's
        own hint can say which value to change.

        Args:
            problem: The failed check.

        Returns:
            The hint, or None to keep the generic one.
        """
        return None
