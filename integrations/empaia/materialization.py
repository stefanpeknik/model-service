"""Materialize external inputs through shared-file links or rectangle mask writers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from integrations.empaia.gateway import EmpaiaGatewayError


if TYPE_CHECKING:
    from pathlib import Path

    from integrations.empaia.http_gateway import (
        DirectAccess,
        Rectangle,
        RectangleMaskWriter,
        SlideMetadata,
    )


def link_direct_input(direct: DirectAccess, destination: Path) -> None:
    """Require supported direct access and an available file, then create a destination symlink; translate filesystem failures into retryable gateway errors.

    Args:
        direct: Provider shared-path metadata and direct-access support flag.
        destination: Local path where input materialization or mask writing must produce
            a file.

    Returns:
        None: No return value.

    Raises:
        EmpaiaGatewayError: AppService transport, authentication, response data, or
            image conversion fails.
    """
    if not direct.supported:
        raise EmpaiaGatewayError(
            "EMPAIA input does not support direct access", retryable=False
        )
    try:
        if not direct.storage_path.is_file():
            raise EmpaiaGatewayError(
                "EMPAIA input is unavailable on shared storage", retryable=True
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(direct.storage_path)
    except OSError as error:
        raise EmpaiaGatewayError(
            "EMPAIA input could not be linked from shared storage", retryable=True
        ) from error


def write_rectangle_input(
    writer: RectangleMaskWriter,
    destination: Path,
    slide: SlideMetadata,
    rectangle: Rectangle,
) -> None:
    """Create the destination directory and invoke the mask writer, preserving classified gateway errors and translating unexpected failures.

    Args:
        writer: Rectangle mask adapter used to create the local region input.
        destination: Local path where input materialization or mask writing must produce
            a file.
        slide: Validated slide dimensions, pyramid levels, and nanometer pixel sizes.
        rectangle: Slide-space region origin and dimensions to rasterize.

    Returns:
        None: No return value.

    Raises:
        EmpaiaGatewayError: Mask geometry or local image materialization fails.
    """
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        writer.write(destination, slide, rectangle)
    except EmpaiaGatewayError:
        raise
    except Exception as error:
        raise EmpaiaGatewayError(
            "EMPAIA rectangle input could not be materialized", retryable=True
        ) from error
