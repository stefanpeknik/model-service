"""Materialize provider inputs in an isolated workspace for one processing attempt.

The public job identity must match protected dispatch context. Each input's
EAD type, recorded at admission, picks the converter that makes its file;
the analysis receives local paths.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from model_service.components import EmpaiaInput
from model_service.components.empaia import converter
from model_service.contracts.failures import (
    BuilderFailure,
    FailureCode,
    InvalidInvocation,
)
from model_service.contracts.invocation import ExecutionContext, ProviderInput
from model_service.directories import make_shared_directory, remove_directory
from model_service.providers.empaia.app_service.gateway import (
    EmpaiaGatewayError,
    EmpaiaInputState,
    EmpaiaJobContext,
)
from model_service.providers.input_access import (
    InputAccess,
    InputAccessFactory,
    close_resource,
)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from model_service.contracts.invocation import Invocation
    from model_service.providers.empaia.app_service.gateway import (
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
        """Retain the workspace, materialized input mapping, and gateway to close after the run.

        Args:
            root: The attempt's workspace.
            gateway: The job's gateway, closed by `aclose`.
            paths: The materialized file of each input, by name.
        """
        self._gateway = gateway
        self._paths = paths
        self._root = root

    def input_path(self, name: str) -> Path:
        """Return the existing materialized file of a declared input.

        Args:
            name: The input name.

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

    async def report_progress(self, fraction: float) -> None:
        """Report the run's progress to the job, for the viewer to show.

        Args:
            fraction: Share of the work done, from 0 to 1.

        Raises:
            EmpaiaGatewayError: The App Service did not take it.
        """
        await self._gateway.report_progress(fraction)

    async def aclose(self) -> None:
        """Close the gateway and delete materialized inputs.

        Outputs stay until provider finish has published them.
        """
        try:
            await self._gateway.close()
        finally:
            await asyncio.to_thread(remove_directory, self._root / "inputs")


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


class EmpaiaInputAccessFactory(InputAccessFactory):
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
    ) -> EmpaiaInputAccess:
        """Materialize every provider input of an attempt in its own workspace.

        The public and protected job contexts must match, and the workspace is named
        by a hash of the job and the attempt ID. On failure the gateway is closed and
        the workspace removed.

        Args:
            invocation: The invocation; its input access state names the job.
            execution_context: The protected job context and processing attempt ID.

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
        # Builder-era jobs recorded no input types; nothing runs them now.
        uses = state.inputs or {}
        slide = next((n for n, u in uses.items() if u.get("type") == "wsi"), None)
        try:
            make_shared_directory(root)
            for name, value in invocation.inputs.items():
                if not isinstance(value, ProviderInput) or name not in uses:
                    raise InvalidInvocation(
                        "EMPAIA input access requires provider inputs"
                    )
                try:
                    found = converter(uses[name])
                except (LookupError, ValidationError) as error:
                    raise InvalidInvocation(
                        f"EMPAIA input {name!r} has no converter"
                    ) from error
                directory = root / "inputs" / name
                make_shared_directory(directory)
                path = await found.materialize(
                    gateway, EmpaiaInput(value.name, slide), directory
                )
                if (
                    not isinstance(path, Path)
                    or not path.is_relative_to(directory)
                    or not path.is_file()
                ):
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
            await asyncio.to_thread(remove_directory, root)
            if isinstance(error, EmpaiaGatewayError):
                # Keep the gateway's safe reason instead of an unexpected failure's.
                raise BuilderFailure(
                    code=FailureCode.INPUT_UNAVAILABLE,
                    safe_message=error.safe_message,
                    retryable=error.retryable,
                ) from error
            raise
        return EmpaiaInputAccess(root, gateway, paths)
