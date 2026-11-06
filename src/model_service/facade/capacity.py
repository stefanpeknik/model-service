"""Tell whether a running execution waits for Ray to find room for its replicas.

Ray Serve keeps a replica it cannot place in `STARTING` with no node, without a
timeout, and calls to it wait. The execution stays `running`; this only
explains why it makes no progress. A replica that has a node but is still
loading is not reported. Answers are best effort: when the Ray dashboard is
unreachable there is no message.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import TYPE_CHECKING

from httpx import HTTPError

from model_service.facade.routes.common import Capacity
from model_service.scheduler.contracts import ExecutionStatus


if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx

    from model_service.scheduler.contracts import ExecutionView


class ServeCapacity(Capacity):
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        origin: str,
        grace_seconds: float = 60,
        cache_seconds: float = 10,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Configure where to read Serve application state and how long to wait.

        Args:
            client: Shared HTTP client, owned by the caller.
            origin: Ray dashboard origin.
            grace_seconds: How long a replica may stay unplaced before it is reported.
            cache_seconds: How long one dashboard answer is reused.
            clock: Returns the current Unix time in seconds.
        """
        self._client = client
        self._url = f"{origin.rstrip('/')}/api/serve/applications/"
        self._grace = grace_seconds
        self._cache_seconds = cache_seconds
        self._clock = clock
        self._applications: dict[str, object] = {}
        self._fetched_at: float | None = None
        self._lock = asyncio.Lock()

    async def explain(self, execution: ExecutionView) -> ExecutionView:
        """Add a message to a running execution whose model or builder cannot be placed.

        Args:
            execution: The execution to explain.

        Returns:
            The execution with a `waiting for capacity` message, or unchanged if
            it is not running, everything is placed, or the dashboard is unreachable.
        """
        if execution.status is not ExecutionStatus.RUNNING:
            return execution
        # Serve application names as deployed by the Helm chart.
        applications = {execution.model_id: f"model {execution.model_id}"}
        if execution.builder_id is not None:
            applications[f"builder-{execution.builder_id}"] = (
                f"builder {execution.builder_id}"
            )
        message = await self._waiting(applications)
        return execution if message is None else replace(execution, message=message)

    async def _waiting(self, applications: dict[str, str]) -> str | None:
        """Describe the first application unplaced for longer than the grace period.

        Args:
            applications: Serve application names mapped to labels for the message.

        Returns:
            The message, or None if no application has waited past the grace period.
        """
        details = await self._details()
        now = self._clock()
        for application, label in applications.items():
            since = _unplaced_since(details.get(application))
            if since is not None and now - since >= self._grace:
                return (
                    f"waiting for capacity: no replica of {label} "
                    f"could be scheduled for {int(now - since)}s"
                )
        return None

    async def _details(self) -> dict[str, object]:
        """Return Serve application details, refetched once the cache expires.

        Returns:
            Application details by name; empty if the last fetch failed.
        """
        async with self._lock:
            now = self._clock()
            if (
                self._fetched_at is None
                or now - self._fetched_at >= self._cache_seconds
            ):
                self._applications = await self._fetch()
                self._fetched_at = now
            return self._applications

    async def _fetch(self) -> dict[str, object]:
        """Read Serve application details from the Ray dashboard.

        Returns:
            Application details by name; empty on any HTTP, status or parse failure.
        """
        try:
            response = await self._client.get(self._url, timeout=5)
            body = response.json() if response.status_code == 200 else None
        except (HTTPError, ValueError):
            return {}
        applications = body.get("applications") if isinstance(body, dict) else None
        return applications if isinstance(applications, dict) else {}


def _unplaced_since(application: object) -> float | None:
    """Find when the oldest unplaced replica started, if a deployment has none running.

    A replica is unplaced when it is `STARTING` with no node.

    Args:
        application: One application's details from the Ray dashboard, unvalidated.

    Returns:
        The oldest unplaced replica's start time in Unix seconds, or None if
        every deployment has a running replica or the data is malformed.
    """
    if not isinstance(application, dict):
        return None
    deployments = application.get("deployments")
    if not isinstance(deployments, dict):
        return None
    oldest: float | None = None
    for deployment in deployments.values():
        replicas = deployment.get("replicas") if isinstance(deployment, dict) else None
        if not isinstance(replicas, list):
            continue
        replicas = [replica for replica in replicas if isinstance(replica, dict)]
        if any(replica.get("state") == "RUNNING" for replica in replicas):
            continue
        for replica in replicas:
            started = replica.get("start_time_s")
            if (
                replica.get("state") == "STARTING"
                and not replica.get("node_id")
                and isinstance(started, int | float)
                and started > 0
            ):
                oldest = started if oldest is None else min(oldest, started)
    return oldest
