"""Turn each declared builder output into its EMPAIA output.

Each builder output's media type (`OutputSpec`) and EMPAIA type (`io`) are
declared up front, so the publisher of each output is known before a job runs:
app generation offers only apps whose outputs all have one, and finish
publishes each output with the publisher of its declaration. Publishing is two
steps: every output is prepared (read, validated, converted) without a write,
and only then are they all committed, so an output EMPAIA cannot take never
leaves the others half published. Supporting a new
EMPAIA type takes a publisher and an entry in `default_publishers`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from model_service.builders.contracts import InlineOutput, SharedArtifact
from model_service.providers.empaia.gateway import EmpaiaGatewayError


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.builders.contracts import BuilderOutput, ModelBinding
    from model_service.builders.spec import OutputSpec
    from model_service.providers.empaia.gateway import EmpaiaGateway


type PublisherKey = tuple[str, str | None]
# Writes one prepared output; nothing before it does.
type Commit = Callable[[], Awaitable[None]]


class OutputPublisher(ABC):
    """Publish one kind of declared output through the gateway's guarded writes."""

    @abstractmethod
    def accepts(self, model: ModelBinding) -> bool:
        """Check whether this model's outputs can be published, e.g. by its channel count.

        Args:
            model: The model whose outputs would be published.

        Returns:
            True if this publisher can publish them.
        """
        ...

    @abstractmethod
    async def prepare(
        self, gateway: EmpaiaGateway, name: str, output: BuilderOutput
    ) -> Commit:
        """Read and validate the output without writing; return its write.

        Args:
            gateway: The job's gateway, written through on commit.
            name: The EMPAIA output key.
            output: The builder output.

        Returns:
            The commit that writes the output.

        Raises:
            EmpaiaGatewayError: If the output cannot be published.
        """
        ...


class PrimitivePublisher(OutputPublisher):
    """Post an inline value as the primitive type its output declares."""

    def __init__(self, output_type: str) -> None:
        """Keep the primitive type to post.

        Args:
            output_type: `bool`, `integer`, `float` or `string`.

        Raises:
            ValueError: If EMPAIA has no such primitive.
        """
        if output_type not in _PRIMITIVE_VALUES:
            raise ValueError(f"EMPAIA has no {output_type!r} primitive")
        self._type = output_type

    def accepts(self, model: ModelBinding) -> bool:
        """Accept every model; a primitive does not depend on it.

        Args:
            model: Unused.

        Returns:
            True.
        """
        return True

    async def prepare(
        self, gateway: EmpaiaGateway, name: str, output: BuilderOutput
    ) -> Commit:
        """Check the inline value fits the primitive type and return its post.

        Args:
            gateway: The job's gateway, posted through on commit.
            name: The EMPAIA output key.
            output: The builder output; must be inline.

        Returns:
            The commit that posts the primitive.

        Raises:
            EmpaiaGatewayError: Not retryable if the output is not inline or its value
                does not fit the type; an empty string does not fit.
        """
        if not isinstance(output, InlineOutput) or not _PRIMITIVE_VALUES[self._type](
            output.value
        ):
            raise EmpaiaGatewayError.invalid(
                f"EMPAIA output {name!r} must be a {self._type} value"
            )
        value = output.value
        return lambda: gateway.post_primitive(name, self._type, value)


# bool is an int in Python, and a float primitive takes whole numbers too.
_PRIMITIVE_VALUES = {
    "bool": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "string": lambda v: isinstance(v, str) and bool(v),
}


class PublisherRegistry:
    """Publishers by the EMPAIA type and media type an output declares."""

    def __init__(self, publishers: Mapping[PublisherKey, OutputPublisher]) -> None:
        """Keep the publishers.

        Args:
            publishers: Publishers by EMPAIA type and media type; the media type is
                None for inline outputs.
        """
        self._publishers = dict(publishers)

    def resolve(
        self, output_type: str | None, spec: OutputSpec
    ) -> OutputPublisher | None:
        """Find the publisher of an output's declared EMPAIA type and media type.

        Args:
            output_type: The output's EMPAIA type; None if `io` declares none.
            spec: The builder's declaration of the output.

        Returns:
            The publisher, or None if none matches.
        """
        if output_type is None:
            return None
        return self._publishers.get((output_type, spec.media_type))

    def publisher_for(
        self,
        name: str,
        output_type: str | None,
        spec: OutputSpec,
        output: BuilderOutput,
    ) -> OutputPublisher:
        """Return the publisher of a produced output, which must be what was declared.

        Args:
            name: The output name.
            output_type: The output's EMPAIA type; None if `io` declares none.
            spec: The builder's declaration of the output.
            output: The produced output.

        Returns:
            The publisher.

        Raises:
            EmpaiaGatewayError: Not retryable if no publisher matches, or the output's
                kind or media type differs from the declaration.
        """
        publisher = self.resolve(output_type, spec)
        if publisher is None:
            raise EmpaiaGatewayError.invalid(f"EMPAIA output {name!r} has no publisher")
        declared = output.kind is spec.kind and (
            not isinstance(output, SharedArtifact)
            or output.media_type == spec.media_type
        )
        if not declared:
            raise EmpaiaGatewayError.invalid(
                f"EMPAIA output {name!r} is not what its builder declares"
            )
        return publisher


def default_publishers() -> PublisherRegistry:
    """Build the publishers of the supported EMPAIA types.

    Inline outputs publish as any primitive type, score grids as continuous
    pixelmaps.

    Returns:
        The registry.
    """
    from model_service.builders.heatmap.score_grid import MEDIA_TYPE as SCORE_GRID
    from model_service.providers.empaia.image_io import ScoreGridPublisher

    publishers: dict[PublisherKey, OutputPublisher] = {
        (output_type, None): PrimitivePublisher(output_type)
        for output_type in _PRIMITIVE_VALUES
    }
    publishers["continuous_pixelmap", SCORE_GRID] = ScoreGridPublisher()
    return PublisherRegistry(publishers)
