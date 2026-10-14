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
    """Require supported direct access and an available file, then create a destination symlink; translate filesystem failures into retryable gateway errors."""
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
    """Create the destination directory and invoke the mask writer, preserving classified gateway errors and translating unexpected failures."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        writer.write(destination, slide, rectangle)
    except EmpaiaGatewayError:
        raise
    except Exception as error:
        raise EmpaiaGatewayError(
            "EMPAIA rectangle input could not be materialized", retryable=True
        ) from error
