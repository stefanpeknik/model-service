from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from workflow_execution.contracts import (
    InvalidWorkflowInvocation,
    ProviderInput,
)


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from execution_scheduler.empaia import EmpaiaGateway, EmpaiaGatewayFactory
    from workflow_execution.contracts import JsonValue, WorkflowInvocation


class EmpaiaInputAccess:
    def __init__(
        self,
        root: Path,
        gateway: EmpaiaGateway,
        paths: dict[str, Path],
    ) -> None:
        self._gateway = gateway
        self._paths = paths
        self._root = root

    def input_path(self, name: str) -> Path:
        try:
            path = self._paths[name]
        except KeyError as error:
            raise InvalidWorkflowInvocation(f"unknown EMPAIA input {name!r}") from error
        if not path.is_file():
            raise InvalidWorkflowInvocation(f"EMPAIA input {name!r} is unavailable")
        return path

    def output_path(self, relative_path: str) -> Path:
        path = self._root / "outputs" / relative_path
        try:
            path.resolve().relative_to(self._root.resolve())
        except ValueError as error:
            raise InvalidWorkflowInvocation(
                "output path escapes workflow storage"
            ) from error
        return path

    def is_visible_artifact(self, path: Path) -> bool:
        try:
            path.resolve(strict=True).relative_to(self._root.resolve(strict=True))
        except (OSError, ValueError):
            return False
        return path.is_file()

    async def aclose(self) -> None:
        await self._gateway.close()


class EmpaiaInputAccessAdapter:
    def __init__(
        self,
        gateway_factory: EmpaiaGatewayFactory,
        workspace_root: Path,
    ) -> None:
        self._gateway_factory = gateway_factory
        self._workspace_root = workspace_root

    async def create(self, invocation: WorkflowInvocation) -> EmpaiaInputAccess:
        state = invocation.input_access.state
        api_url = _state_string(state, "api_url")
        job_id = _state_string(state, "job_id")
        credential = _state_string(state, "credential")
        processing_attempt_id = _state_string(state, "processing_attempt_id")
        gateway = self._gateway_factory(api_url, job_id, credential)
        execution_key = hashlib.sha256(
            f"{api_url}\0{job_id}\0{processing_attempt_id}".encode()
        ).hexdigest()
        root = self._workspace_root / execution_key
        root.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        try:
            for name, value in invocation.inputs.items():
                if not isinstance(value, ProviderInput):
                    raise InvalidWorkflowInvocation(
                        "EMPAIA input access requires provider inputs"
                    )
                path = root / "inputs" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                await gateway.materialize_input(value.name, str(path))
                if not path.is_file():
                    raise InvalidWorkflowInvocation(
                        f"EMPAIA input {name!r} was not materialized"
                    )
                paths[name] = path
        except Exception:
            await gateway.close()
            raise
        return EmpaiaInputAccess(root, gateway, paths)


def _state_string(state: Mapping[str, JsonValue], name: str) -> str:
    value = state.get(name)
    if not isinstance(value, str) or not value:
        raise InvalidWorkflowInvocation(
            f"EMPAIA input access requires a non-empty {name}"
        )
    return value
