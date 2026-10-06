from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from httpx import HTTPError


if TYPE_CHECKING:
    from collections.abc import Mapping


class RouteResponse(Protocol):
    status_code: int

    def json(self) -> object: ...


class RouteClient(Protocol):
    async def get(self, url: str, *, timeout: float) -> RouteResponse: ...


class WorkflowDeploymentState(StrEnum):
    READY = "ready"
    UNREACHABLE = "unreachable"
    MISSING_APPLICATIONS = "missing_applications"
    NOT_CONFIGURED = "not_configured"


@dataclass(frozen=True, slots=True)
class WorkflowDeploymentStatus:
    status: WorkflowDeploymentState
    missing_workflows: tuple[str, ...] = ()


class WorkflowDeploymentHealth:
    def __init__(
        self,
        client: RouteClient,
        *,
        origin: str,
        workflow_ids: set[str] | frozenset[str],
    ) -> None:
        self._client = client
        self._origin = origin.rstrip("/")
        self._workflow_ids = frozenset(workflow_ids)

    async def check(self) -> WorkflowDeploymentStatus:
        try:
            response = await self._client.get(
                f"{self._origin}/-/routes",
                timeout=5,
            )
            if response.status_code != 200:
                return WorkflowDeploymentStatus(WorkflowDeploymentState.UNREACHABLE)
            routes = response.json()
        except (HTTPError, TypeError, ValueError):
            return WorkflowDeploymentStatus(WorkflowDeploymentState.UNREACHABLE)
        if not isinstance(routes, dict):
            return WorkflowDeploymentStatus(WorkflowDeploymentState.UNREACHABLE)
        route_names = _route_names(routes)
        missing = tuple(
            workflow_id
            for workflow_id in sorted(self._workflow_ids)
            if f"/workflows/{workflow_id}" not in route_names
        )
        return WorkflowDeploymentStatus(
            WorkflowDeploymentState.MISSING_APPLICATIONS
            if missing
            else WorkflowDeploymentState.READY,
            missing,
        )


def _route_names(routes: Mapping[object, object]) -> set[str]:
    names: set[str] = set()
    for key, value in routes.items():
        if isinstance(key, str):
            names.add(key.rstrip("/") or "/")
        if isinstance(value, str) and value.startswith("/"):
            names.add(value.rstrip("/") or "/")
    return names
