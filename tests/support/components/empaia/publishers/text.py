from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from components.base import Commit, EmpaiaOutput, Publisher


if TYPE_CHECKING:
    from pathlib import Path


class Text(Publisher):
    """Test publisher: shares its name with the `text` converter."""

    name = "text"
    reads_format = "score_grid"
    emits = frozenset({"string"})
    example_options: ClassVar[dict[str, Any]] = {"type": "string"}

    def io_fragment(self, item: EmpaiaOutput) -> dict[str, Any]:
        """Declare a string output."""
        return {"type": "string"}

    async def prepare(
        self, gateway: Any, item: EmpaiaOutput, artifact: Path, media_type: str
    ) -> Commit:
        """Nothing to check; the commit posts the output."""

        async def commit() -> None:
            await gateway.post_output(item.name, "text")

        return commit
