"""The `my_component` publisher: a `<data format>` output as an EMPAIA `<EAD type>`.

Copy to `src/components/providers/empaia/publishers/<name>.py` and replace
`my_component`, `MyComponent` and the placeholders in <angle brackets>. A
format with no sample yet needs a writer in `SAMPLES` in
`tests/components/helpers/empaia.py`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, ClassVar

from components import Commit, EmpaiaOutput, Publisher
from model_service.formats import value


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.providers.empaia.app_service.gateway import EmpaiaGateway


class MyComponent(Publisher):
    """<One sentence: what it publishes, as which EMPAIA type>.

    `outputs: {<output>: {type: string}}` under `providers.empaia`.
    """

    name = "my_component"
    reads_format = value.FORMAT.name
    emits = frozenset({"string"})
    example_options: ClassVar[dict[str, Any]] = {"type": "string"}

    def io_fragment(self, item: EmpaiaOutput) -> dict[str, Any]:
        """The EAD `io` entry of the output.

        Args:
            item: The output.

        Returns:
            The entry.
        """
        return {"type": self.options.type}

    async def prepare(
        self,
        gateway: EmpaiaGateway,
        item: EmpaiaOutput,
        artifact: Path,
        media_type: str,
    ) -> Commit:
        """Read and check the output; write nothing yet.

        Args:
            gateway: The job's gateway; written through only on commit.
            item: The output.
            artifact: The stored output file.
            media_type: Its media type.

        Returns:
            The commit that writes it.
        """
        scalar = await asyncio.to_thread(value.read, artifact, media_type)

        async def commit() -> None:
            await gateway.post_output(item.name, {"value": str(scalar)})

        return commit
