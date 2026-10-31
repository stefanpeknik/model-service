"""Materialize external inputs through shared-file links or rectangle mask writers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.providers.empaia.gateway import EmpaiaGatewayError


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.providers.empaia.http_gateway import (
        DirectAccess,
        Rectangle,
        RectangleMaskWriter,
        SlideMetadata,
    )


def link_direct_input(direct: DirectAccess, directory: Path) -> Path:
    """Link the shared file into `directory` under its own name and return the link.

    The name is kept because slide readers pick the format by extension, and a
    MIRAX `.mrxs` file is linked together with its same-named data directory.
    Filesystem failures become retryable gateway errors.
    """
    if not direct.supported:
        raise EmpaiaGatewayError(
            "EMPAIA input does not support direct access", retryable=False
        )
    source = direct.storage_path
    try:
        if not source.is_file():
            raise EmpaiaGatewayError(
                "EMPAIA input is unavailable on shared storage", retryable=True
            )
        directory.mkdir(parents=True, exist_ok=True)
        link = directory / source.name
        link.symlink_to(source)
        companion = source.with_suffix("")
        if source.suffix.lower() == ".mrxs" and companion.is_dir():
            (directory / companion.name).symlink_to(companion, target_is_directory=True)
        return link
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
