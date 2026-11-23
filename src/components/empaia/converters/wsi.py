"""`wsi`: the job's slide, linked from the installation's slide volume."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from components.base import EmpaiaInput, InputConverter
from model_service.providers.empaia.app_service.gateway import EmpaiaGatewayError


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.providers.empaia.app_service.gateway import EmpaiaGateway


class Wsi(InputConverter):
    """A slide input: the file it is stored as, linked under its own name.

    The RationAI direct-access route names the file on the slide volume,
    which model-service mounts where the installation's CDS does. A MIRAX
    `.mrxs` comes with its data directory. A slide the route has no path for
    is refused: model-service does not download slides.
    """

    name = "wsi"
    ead_type = "wsi"

    def io_fragment(self, item: EmpaiaInput) -> dict[str, Any]:
        """Declare a slide.

        Args:
            item: The input.

        Returns:
            `{"type": "wsi"}`.
        """
        return {"type": "wsi"}

    async def materialize(
        self, gateway: EmpaiaGateway, item: EmpaiaInput, directory: Path
    ) -> Path:
        """Link the slide's file into `directory`.

        Args:
            gateway: The job's gateway.
            item: The input.
            directory: Where to link it.

        Returns:
            The link.

        Raises:
            EmpaiaGatewayError: Not retryable if the slide is not on the slide
                volume or does not support direct access; retryable if the
                file is missing or linking fails.
        """
        from model_service.providers.empaia.app_service.input_files import (
            link_direct_input,
        )

        direct = await gateway.direct_access(item.name)
        if direct is None:
            raise EmpaiaGatewayError.invalid(
                "EMPAIA slide is not on the slide volume model-service reads"
            )
        return link_direct_input(direct, directory)
