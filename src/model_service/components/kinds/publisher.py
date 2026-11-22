"""Publisher: carries a stored output into EMPAIA via prepare, then commit."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from model_service.components.component import Component
from model_service.components.options import Options
from model_service.components.slots import EMPAIA_OUTPUTS


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from model_service.components.columns import Col
    from model_service.components.problems import Problem
    from model_service.components.slots import Slot


type Commit = Callable[[], Awaitable[None]]
"""A prepared publication: everything is checked, only the write remains."""


class PublisherOptions(Options):
    """Options every publisher has.

    Attributes:
        type: The EAD output type, from the analysis's `expose.empaia`.
    """

    type: str = Field(
        description="The EAD output type to publish as, e.g. `continuous_pixelmap`."
    )


@dataclass(frozen=True)
class EmpaiaOutput:
    """One output of an EMPAIA app, as a publisher sees it.

    Attributes:
        name: The EAD io key, the sink's output name.
        slide: The io key of the app's `wsi` input, which the output
            references; None if the app has none.
    """

    name: str
    slide: str | None = None


class Publisher(Component):
    """Publishes a stored output of one data format as an EMPAIA output type.

    The analysis's `expose.empaia` names the EAD type of each output; the
    publisher that reads the sink's format and emits that type is picked.

    Attributes:
        reads_format: The data format of the output it publishes.
        emits: The EAD types it can publish as.
    """

    kind: ClassVar[str] = "publisher"
    slots: ClassVar[tuple[Slot, ...]] = (EMPAIA_OUTPUTS,)
    Options: ClassVar[type[Options]] = PublisherOptions
    reads_format: ClassVar[str]
    emits: ClassVar[frozenset[str]]
    options: PublisherOptions

    @abstractmethod
    def io_fragment(self, item: EmpaiaOutput) -> dict[str, Any]:
        """The EAD `io` entry of the output.

        Args:
            item: The output.

        Returns:
            The entry, e.g. `{"type": "continuous_pixelmap", ...}`.
        """

    def check(self, columns: Mapping[str, Col]) -> Problem | None:
        """Check, at load, that what the sink writes fits the EAD type.

        Args:
            columns: The columns that reach the sinks, every dimension known.

        Returns:
            What doesn't fit, or None.
        """
        return None

    @abstractmethod
    async def prepare(
        self, gateway: Any, item: EmpaiaOutput, artifact: Path, media_type: str
    ) -> Commit:
        """Read and check everything, and write nothing.

        Args:
            gateway: The job's `EmpaiaGateway`; read now, written on commit.
            item: The output.
            artifact: The stored output.
            media_type: Its stored media type.

        Returns:
            The commit that writes it.

        Raises:
            EmpaiaGatewayError: With a safe message, if it cannot be published.
        """
