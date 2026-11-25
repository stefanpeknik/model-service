"""Link a job's slide into its workspace."""

from __future__ import annotations

from pathlib import Path

from model_service.directories import make_shared_directory
from model_service.providers.empaia.app_service.gateway import EmpaiaGatewayError
from model_service.providers.empaia.app_service.types import DirectAccess


def link_direct_input(direct: DirectAccess, directory: Path) -> Path:
    """Link the shared file into `directory` under its own name and return the link.

    The name is kept because slide readers pick the format by extension, and a
    MIRAX `.mrxs` file is linked together with its same-named data directory.

    Args:
        direct: The shared file and whether direct access is supported.
        directory: The directory to link into; created if missing.

    Returns:
        The link to the file.

    Raises:
        EmpaiaGatewayError: Not retryable if the input does not support direct
            access; retryable if the file is missing or linking fails.
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
        make_shared_directory(directory)
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
