"""`primitive`: a `value` output as an EMPAIA primitive (bool, integer, float or string)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, ClassVar

from model_service.components import Commit, EmpaiaOutput, Publisher
from model_service.formats import value
from model_service.providers.empaia.app_service.gateway import (
    EmpaiaGatewayError,
    PrimitiveType,
)


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.providers.empaia.app_service.gateway import EmpaiaGateway


class Primitive(Publisher):
    """Posts one scalar as the primitive type the analysis names.

    The value must fit the type: a bool is not an integer, a float takes
    whole numbers, and an empty string is refused. It is checked before
    anything is posted.
    """

    name = "primitive"
    reads_format = value.NAME
    emits = frozenset(str(t) for t in PrimitiveType)
    example_options: ClassVar[dict[str, Any]] = {"type": "float"}

    def io_fragment(self, item: EmpaiaOutput) -> dict[str, Any]:
        """Declare a primitive of the named type.

        Args:
            item: The output.

        Returns:
            E.g. `{"type": "float"}`.
        """
        return {"type": self.options.type}

    async def prepare(
        self,
        gateway: EmpaiaGateway,
        item: EmpaiaOutput,
        artifact: Path,
        media_type: str,
    ) -> Commit:
        """Read the value, check it fits, and build the body to post.

        Args:
            gateway: The job's gateway, posted through on commit.
            item: The output.
            artifact: The stored `value` file.
            media_type: Its stored media type.

        Returns:
            The commit that posts the primitive.

        Raises:
            EmpaiaGatewayError: Not retryable if the file is unreadable or the
                value does not fit the type.
        """
        kind = PrimitiveType(self.options.type)
        try:
            scalar = await asyncio.to_thread(value.read, artifact, media_type)
        except (OSError, ValueError) as error:
            raise EmpaiaGatewayError.invalid(
                f"EMPAIA output {item.name!r} is not a readable value"
            ) from error
        if not kind.fits(scalar):
            raise EmpaiaGatewayError.invalid(
                f"EMPAIA output {item.name!r} must be a {kind} value"
            )
        body = primitive_body(item.name, gateway.job_id, kind, scalar)

        async def commit() -> None:
            """Post the prepared primitive."""
            await gateway.post_output(item.name, body)

        return commit


def primitive_body(
    name: str, job_id: str, kind: PrimitiveType, scalar: value.Scalar
) -> Any:
    """The SDK's request model of a primitive output, created by the job.

    Args:
        name: The EMPAIA output key.
        job_id: The job, recorded as the creator.
        kind: The primitive type.
        scalar: A value that fits it.

    Returns:
        The body to post.
    """
    from rationai.empaia.typing import primitives

    model = {
        PrimitiveType.BOOL: primitives.PostBoolPrimitive,
        PrimitiveType.INTEGER: primitives.PostIntegerPrimitive,
        PrimitiveType.FLOAT: primitives.PostFloatPrimitive,
        PrimitiveType.STRING: primitives.PostStringPrimitive,
    }[kind]
    return model.model_validate(
        {
            "name": name,
            "type": str(kind),
            "value": scalar,
            "creator_id": job_id,
            "creator_type": "job",
        }
    )
