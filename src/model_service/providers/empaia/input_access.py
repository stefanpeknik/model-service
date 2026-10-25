"""Materialize provider inputs in an isolated workflow workspace for one processing attempt.

The public job identity must match protected dispatch context. The workflow receives
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


class EmpaiaInputAccess:
    """Expose materialized job inputs and completed artifacts within one attempt's workspace."""

    def __init__(
        self,
        root: Path,
        gateway: EmpaiaGateway,
        paths: dict[str, Path],
    ) -> None:
        """Retain the workspace, materialized input mapping, and gateway to close after workflow execution."""
        self._gateway = gateway
        self._paths = paths
        self._root = root

    def input_path(self, name: str) -> Path:
        """Return an existing materialized file for a declared input or raise an invocation error."""
        try:
            path = self._paths[name]
        except KeyError as error:
            raise InvalidInvocation(f"unknown EMPAIA input {name!r}") from error
        if not path.is_file():
            raise InvalidInvocation(f"EMPAIA input {name!r} is unavailable")
        return path

    def output_path(self, relative_path: str) -> Path:
        """Resolve an output under the workspace and reject targets that escape its root."""
        path = self._root / "outputs" / relative_path
        try:
            path.resolve().relative_to(self._root.resolve())
        except ValueError as error:
            raise InvalidInvocation("output path escapes workflow storage") from error
        return path

    def is_visible_artifact(self, path: Path) -> bool:
        """Require an existing file contained within the resolved attempt workspace."""
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


class EmpaiaInputAccessAdapter:
    """Create per-attempt local access by resolving inputs through an EMPAIA gateway."""

    def __init__(
        self,
        gateway_factory: EmpaiaGatewayFactory,
        workspace_root: Path,
    ) -> None:
        """Retain the gateway factory and parent directory for isolated processing workspaces."""
        self._gateway_factory = gateway_factory
        self._workspace_root = workspace_root

    async def create(
        self,
        invocation: Invocation,
        execution_context: ExecutionContext,
        model: ModelBinding | None = None,
    ) -> EmpaiaInputAccess:
        """Validate matching job contexts and an attempt ID, hash the workspace identity, and materialize every provider input. Close acquired resources if creation fails."""
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
        execution_key = hashlib.sha256(
            f"{context.api_url}\0{context.job_id}\0{processing_attempt_id}".encode()
        ).hexdigest()
        root = self._workspace_root / execution_key
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
                path = Path(
                    await gateway.materialize_input(
                        value.name,
                        str(directory),
                        mpp=model.mpp if model is not None else None,
                    )
                )
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
            if isinstance(error, EmpaiaGatewayError):
                # Keep the gateway's safe reason instead of "builder failed unexpectedly".
                raise BuilderFailure(
                    code=FailureCode.INPUT_UNAVAILABLE,
                    safe_message=error.safe_message,
                    retryable=error.retryable,
                ) from error
            raise
        return EmpaiaInputAccess(root, gateway, paths)
