"""Check that Ray Serve answers and exposes every expected builder and model route.

Route presence says the applications are deployed, not that a GPU is warm.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from httpx import HTTPError


if TYPE_CHECKING:
    from collections.abc import Iterable

    import httpx


class DeploymentState(StrEnum):
    READY = "ready"
    UNREACHABLE = "unreachable"
    MISSING_APPLICATIONS = "missing_applications"
    NOT_CONFIGURED = "not_configured"


@dataclass(frozen=True, slots=True)
class DeploymentStatus:
    status: DeploymentState
    missing_routes: tuple[str, ...] = ()


class DeploymentHealthCheck(ABC):
    @abstractmethod
    async def check(self) -> DeploymentStatus:
        """Report whether Ray Serve exposes every expected builder and model route.

        Returns:
            The deployment state and any missing routes.
        """
        ...


class DeploymentHealth(DeploymentHealthCheck):
    def __init__(
        self, client: httpx.AsyncClient, *, origin: str, routes: Iterable[str]
    ) -> None:
        """Configure the Ray Serve origin and the routes that must be deployed.

        Args:
            client: Shared HTTP client, owned by the caller.
            origin: Ray Serve origin.
            routes: Route prefixes every builder and model must expose.
        """
        self._client = client
        self._origin = origin.rstrip("/")
        self._routes = tuple(sorted(set(routes)))

    async def check(self) -> DeploymentStatus:
        """Compare Ray Serve's `/-/routes` with the expected routes.

        Returns:
            `UNREACHABLE` if Serve does not answer with a route map,
            `MISSING_APPLICATIONS` with the missing routes, or `READY`.
        """
        try:
            response = await self._client.get(f"{self._origin}/-/routes", timeout=5)
            routes = response.json() if response.status_code == 200 else None
        except (HTTPError, ValueError):
            routes = None
        if not isinstance(routes, dict):
            return DeploymentStatus(DeploymentState.UNREACHABLE)
        # Ray returns {route_prefix: application_name}.
        deployed = {str(prefix).rstrip("/") or "/" for prefix in routes}
        missing = tuple(route for route in self._routes if route not in deployed)
        return DeploymentStatus(
            DeploymentState.MISSING_APPLICATIONS if missing else DeploymentState.READY,
            missing,
        )
