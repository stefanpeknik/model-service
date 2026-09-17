from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from collections.abc import Mapping


class RouteResponse(Protocol):
    status_code: int

    def json(self) -> object: ...


class RouteClient(Protocol):
    async def get(self, url: str, *, timeout: float) -> RouteResponse: ...


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

    async def missing_workflows(self) -> tuple[str, ...]:
        try:
            response = await self._client.get(
                f"{self._origin}/-/routes",
                timeout=5,
            )
            if response.status_code != 200:
                return tuple(sorted(self._workflow_ids))
            routes = response.json()
        except Exception:  # noqa: BLE001 - readiness must fail closed
            return tuple(sorted(self._workflow_ids))
        if not isinstance(routes, dict):
            return tuple(sorted(self._workflow_ids))
        route_names = _route_names(routes)
        return tuple(
            workflow_id
            for workflow_id in sorted(self._workflow_ids)
            if f"/workflows/{workflow_id}" not in route_names
        )


def _route_names(routes: Mapping[object, object]) -> set[str]:
    names: set[str] = set()
    for key, value in routes.items():
        if isinstance(key, str):
            names.add(key.rstrip("/") or "/")
        if isinstance(value, str) and value.startswith("/"):
            names.add(value.rstrip("/") or "/")
    return names
