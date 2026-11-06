"""Materialize provider inputs in an isolated workspace for one processing attempt.

The public job identity must match protected dispatch context. The builder receives
local paths while the gateway owns provider communication.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from model_service.builders.application import AccessAdapter, InputAccess
from model_service.builders.cleanup import close_resource
from model_service.builders.contracts import (
    BuilderFailure,
    ExecutionContext,
    FailureCode,
    InvalidInvocation,
    ProviderInput,
)
from model_service.providers.empaia.gateway import (
    EmpaiaGatewayError,
    EmpaiaInputState,
    EmpaiaJobContext,
)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from model_service.builders.contracts import Invocation, ModelBinding
    from model_service.providers.empaia.gateway import (
        EmpaiaGateway,
        EmpaiaGatewayFactory,
    )


class EmpaiaInputAccess(InputAccess):
    """Expose materialized job inputs and completed artifacts within one attempt's workspace."""

    def __init__(
        self,
        root: Path,
        gateway: EmpaiaGateway,
        paths: dict[str, Path],
    ) -> None:
        """Retain the workspace, materialized input mapping, and gateway to close after the builder ran.

        Args:
            root: The attempt's workspace.
            gateway: The job's gateway, closed by `aclose`.
            paths: The materialized file of each builder input, by name.
        """
        self._gateway = gateway
        self._paths = paths
        self._root = root

    def input_path(self, name: str) -> Path:
        """Return the existing materialized file of a declared input.

        Args:
            name: The builder input name.

        Returns:
            The file path.

        Raises:
            InvalidInvocation: If the input is unknown or its file is missing.
        """
        try:
            path = self._paths[name]
        except KeyError as error:
            raise InvalidInvocation(f"unknown EMPAIA input {name!r}") from error
        if not path.is_file():
            raise InvalidInvocation(f"EMPAIA input {name!r} is unavailable")
        return path

    def output_path(self, relative_path: str) -> Path:
        """Resolve an output below the workspace's `outputs` and reject targets that escape the root.

        Args:
            relative_path: The path below `outputs`.

        Returns:
            The output path; it need not exist.

        Raises:
            InvalidInvocation: If the path escapes the attempt workspace.
        """
        path = self._root / "outputs" / relative_path
        try:
            path.resolve().relative_to(self._root.resolve())
        except ValueError as error:
            raise InvalidInvocation(
                "output path escapes the attempt workspace"
            ) from error
        return path

    def is_visible_artifact(self, path: Path) -> bool:
        """Check that a path is an existing file within the resolved attempt workspace.

        Args:
            path: The artifact path a builder reported.

        Returns:
            True if the path is a file inside the workspace.
        """
        try:
            path.resolve(strict=True).relative_to(self._root.resolve(strict=True))
        except (OSError, ValueError):
            return False
        return path.is_file()

    async def aclose(self) -> None:
        """Close the gateway and delete materialized inputs.

        Outputs stay until provider finish has published them.
        """
        try:
            await self._gateway.close()
        finally:
            await asyncio.to_thread(shutil.rmtree, self._root / "inputs", True)


def job_workspace(root: Path, api_url: str, job_id: str) -> Path:
    """Return the directory of all attempt workspaces of one EMPAIA job; provider finish removes it.

    Args:
        root: The workspace root.
        api_url: The job's App Service URL.
        job_id: The EMPAIA job ID.

    Returns:
        `root` joined with a SHA-256 of the API URL and job ID.
    """
    return root / hashlib.sha256(f"{api_url}\0{job_id}".encode()).hexdigest()


class EmpaiaInputAccessAdapter(AccessAdapter):
    """Create per-attempt local access by resolving inputs through an EMPAIA gateway."""

    def __init__(
        self,
        gateway_factory: EmpaiaGatewayFactory,
        workspace_root: Path,
    ) -> None:
        """Retain the gateway factory and parent directory for isolated processing workspaces.

        Args:
            gateway_factory: Creates the gateway of each job.
            workspace_root: The parent of the job workspaces.
        """
        self._gateway_factory = gateway_factory
        self._workspace_root = workspace_root

    async def create(
        self,
        invocation: Invocation,
        execution_context: ExecutionContext,
        model: ModelBinding | None = None,
    ) -> EmpaiaInputAccess:
        """Materialize every provider input of an attempt in its own workspace.

        The public and protected job contexts must match, and the workspace is named
        by a hash of the job and the attempt ID. On failure the gateway is closed and
        the workspace removed.

        Args:
            invocation: The invocation; its input access state names the job.
            execution_context: The protected job context and processing attempt ID.
            model: Unused.

        Returns:
            Access to the materialized inputs and the attempt's outputs.

        Raises:
            InvalidInvocation: If a context is malformed, the contexts name different
                jobs, there is no attempt ID, an input is not a provider input, or the
                gateway returns a path outside the input's directory or not a file.
            BuilderFailure: With `INPUT_UNAVAILABLE` and the gateway's safe message and
                retry flag, if materializing raises `EmpaiaGatewayError`.
        """
        try:
            state = EmpaiaInputState.model_validate(invocation.input_access.state)
            context = EmpaiaJobContext.model_validate(
                execution_context.protected_context
            )
        except ValidationError as error:
            raise InvalidInvocation("EMPAIA input access context is invalid") from error
        if state.api_url != context.api_url or state.job_id != context.job_id:
            raise InvalidInvocation("EMPAIA public and protected job contexts differ")
        processing_attempt_id = execution_context.processing_attempt_id
        if processing_attempt_id is None:
            raise InvalidInvocation("EMPAIA input access requires a processing attempt")
        gateway = self._gateway_factory(context)
        root = (
            job_workspace(self._workspace_root, context.api_url, context.job_id)
            / hashlib.sha256(processing_attempt_id.encode()).hexdigest()
        )
        paths: dict[str, Path] = {}
        try:
            root.mkdir(parents=True, exist_ok=True)
            for name, value in invocation.inputs.items():
                if not isinstance(value, ProviderInput):
                    raise InvalidInvocation(
                        "EMPAIA input access requires provider inputs"
                    )
                directory = root / "inputs" / name
                directory.mkdir(parents=True, exist_ok=True)
                path = Path(await gateway.materialize_input(value.name, str(directory)))
                if not path.is_relative_to(directory) or not path.is_file():
                    raise InvalidInvocation(
                        f"EMPAIA input {name!r} was not materialized"
                    )
                paths[name] = path
        except BaseException as error:
            try:
                await close_resource(gateway.close)
            except Exception as cleanup_error:
                logger.warning(
                    "EMPAIA input cleanup failed: %s", type(cleanup_error).__name__
                )
            await asyncio.to_thread(shutil.rmtree, root, True)
            if isinstance(error, EmpaiaGatewayError):
                # Keep the gateway's safe reason instead of "builder failed unexpectedly".
                raise BuilderFailure(
                    code=FailureCode.INPUT_UNAVAILABLE,
                    safe_message=error.safe_message,
                    retryable=error.retryable,
                ) from error
            raise
        return EmpaiaInputAccess(root, gateway, paths)
