"""The `my_component` step: <what it does to each item>.

Copy to `src/components/steps/<name>.py` and replace
`my_component`, `MyComponent` and the placeholders in <angle brackets>.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from components.base import Col, Options, Step


if TYPE_CHECKING:
    from components.base import Batch


class MyComponentOptions(Options):
    """Options of the `my_component` step.

    Attributes:
        factor: <What the option changes>.
    """

    factor: float = Field(1.0, gt=0, description="<What the option changes>.")


class MyComponent(Step):
    """<One sentence: what it does and where to use it>.

    `post: [{use: my_component}]` in a served model, or a step of an analysis.
    """

    name = "my_component"
    Options = MyComponentOptions
    example_options: ClassVar[dict[str, Any]] = {"factor": 1.0}
    reads: ClassVar[dict[str, Col]] = {"output": Col("float32", None)}
    adds: ClassVar[dict[str, Col]] = {"output": Col("float32", None)}
    options: MyComponentOptions

    def __call__(self, batch: Batch) -> Batch:
        """Transform each row; keep every row.

        Args:
            batch: Rows with the declared `reads`.

        Returns:
            The batch with the declared `adds`.
        """
        return batch | {"output": batch["output"] * self.options.factor}
