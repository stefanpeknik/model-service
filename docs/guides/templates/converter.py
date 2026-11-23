"""The `my_component` converter: an EMPAIA `<EAD type>` input as a run input.

Copy to `src/model_service/components/empaia/input_converters/<name>.py` and
replace `my_component`, `MyComponent` and the placeholders in <angle brackets>.
Add example data for the EAD type to `EXAMPLE_INPUTS` in
`src/model_service/testing/empaia.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from model_service.components import EmpaiaInput, InputConverter


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.contracts.json_values import JsonValue
    from model_service.providers.empaia.app_service.gateway import EmpaiaGateway


class MyComponent(InputConverter):
    """<One sentence: which EMPAIA input it converts, into what>.

    `inputs: {<input>: {type: string}}` under `expose.empaia`.
    """

    name = "my_component"
    ead_type = "string"
    gives_file = False

    def io_fragment(self, item: EmpaiaInput) -> dict[str, Any]:
        """The EAD `io` entry of the input.

        Args:
            item: The input.

        Returns:
            The entry.
        """
        return {"type": self.ead_type}

    async def materialize(
        self, gateway: EmpaiaGateway, item: EmpaiaInput, directory: Path
    ) -> Path | JsonValue:
        """Read the job's input; a file goes into `directory`.

        Args:
            gateway: The job's gateway; only read through it.
            item: The input.
            directory: An empty directory for a file.

        Returns:
            The value, or the file when `gives_file` is True.
        """
        return str(await gateway.get_input_data(item.name))
