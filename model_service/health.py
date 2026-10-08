"""Inspect Ray Serve routes without coupling compute health to durable API readiness.

Route presence is a deployment availability check, not a model prediction or GPU health
test.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from httpx import HTTPError


if TYPE_CHECKING:
    from collections.abc import Mapping


class RouteResponse(Protocol):
    """Specify the HTTP status and JSON body required to inspect Ray's route listing.

    Attributes:
        status_code: HTTP status of the upstream response.
    """

    status_code: int

    def json(self) -> object:
        """Decode the upstream route listing response.

        Returns:
            object: Decoded Ray Serve route-listing JSON.

        Raises:
            ValueError: The HTTP response body cannot be decoded as JSON.
        """
        ...


class RouteClient(Protocol):
    """Define the short HTTP GET used by the deployment health check."""

    async def get(self, url: str, *, timeout: float) -> RouteResponse:
        """Fetch the route listing with an explicit timeout.

        Args:
            url: Complete upstream HTTP request URL.
            timeout: Request or scenario time budget in seconds.

        Returns:
            RouteResponse: HTTP response containing the deployment route listing.
        """
        ...


class WorkflowDeploymentState(StrEnum):
    """Distinguish ready compute, unreachable Ray, missing routes, and absent health configuration."""

    READY = "ready"
    UNREACHABLE = "unreachable"
    MISSING_APPLICATIONS = "missing_applications"
    NOT_CONFIGURED = "not_configured"


@dataclass(frozen=True, slots=True)
class WorkflowDeploymentStatus:
    """Return deployment state and the IDs of missing workflow applications.

    Attributes:
        status: Ray workflow deployment availability classification.
        missing_workflows: Declared workflow IDs absent from Ray Serve routing.
    """

    status: WorkflowDeploymentState
    missing_workflows: tuple[str, ...] = ()


class WorkflowDeploymentHealth:
    """Check that Ray Serve responds and exposes routes for all declared workflows."""

    def __init__(
        self,
        client: RouteClient,
        *,
        origin: str,
        workflow_ids: set[str] | frozenset[str],
    ) -> None:
        """Retain the caller-owned client, normalized Ray origin, and required workflow IDs.

        Args:
            client: Caller-owned HTTP client; this adapter does not close it.
            origin: Upstream service HTTP origin, normalized without a trailing slash.
            workflow_ids: Workflow identifiers whose Ray HTTP routes must be present.
        """
        self._client = client
        self._origin = origin.rstrip("/")
        self._workflow_ids = frozenset(workflow_ids)

    async def check(self) -> WorkflowDeploymentStatus:
        """Fetch Ray's route listing and return a safe state for transport, decoding, or missing-route failures.

        Returns:
            WorkflowDeploymentStatus: Ray route availability and any missing workflow
                IDs.
        """
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
    """Normalize route paths from either keys or path-valued entries, removing trailing slashes.

    Args:
        routes: Decoded Ray Serve route listing.

    Returns:
        set[str]: Normalized route paths found in either keys or path-valued listing
            entries.
    """
    names: set[str] = set()
    for key, value in routes.items():
        if isinstance(key, str):
            names.add(key.rstrip("/") or "/")
        if isinstance(value, str) and value.startswith("/"):
            names.add(value.rstrip("/") or "/")
    return names
