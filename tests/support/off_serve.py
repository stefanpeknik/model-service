"""Run a Ray Serve model deployment in this process, for equivalence tests.

The deployment's own class is built and reconfigured without Serve. Its
FastAPI routes are served by a test client, with the instance injected where
Serve would inject the replica. `@serve.batch` methods need a replica, so each
is replaced on the instance by a call of the undecorated method with a batch
of one; `batch(method, items)` runs the undecorated method on a whole batch.

Deleted with the old model classes it exists to compare.
"""

from __future__ import annotations

import asyncio
import inspect
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from fastapi.testclient import TestClient


if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastapi import FastAPI
    from httpx import Response


class OffServe:
    """A reconfigured deployment instance with its routes on a test client."""

    def __init__(self, instance: Any, client: TestClient) -> None:
        self.instance = instance
        self.client = client

    def call(self, method: str, *args: Any) -> Any:
        """Call a handle method on the client's event loop."""
        result = getattr(self.instance, method)(*args)
        if inspect.isawaitable(result):
            return self.client.portal.call(_awaited, result)
        return result

    def gather(self, method: str, items: list[Any]) -> list[Any]:
        """Call a handle method once per item, all at once."""

        async def calls() -> list[Any]:
            bound = getattr(self.instance, method)
            return list(await asyncio.gather(*(bound(item) for item in items)))

        return self.client.portal.call(calls)

    def batch(self, method: str, items: list[Any]) -> list[Any]:
        """Run an `@serve.batch` method's undecorated body on one batch."""
        body = getattr(type(self.instance), method).__wrapped__
        return self.client.portal.call(body, self.instance, items)

    def post(self, body: bytes, headers: dict[str, str] | None = None) -> Response:
        """POST to the deployment's raw-inference route."""
        return self.client.post("/", content=body, headers=headers or {})


@contextmanager
def off_serve(
    deployment: Any, app: FastAPI, config: dict[str, Any]
) -> Iterator[OffServe]:
    """Build, reconfigure and serve a deployment class in this process.

    Args:
        deployment: The `@serve.deployment` object.
        app: The FastAPI app its `@serve.ingress` registered routes on.
        config: Its `user_config`.

    Yields:
        The running instance.
    """
    cls = deployment.func_or_class.__wrapped__
    instance = cls()
    instance.reconfigure(config)
    for name, member in inspect.getmembers(cls):
        if hasattr(member, "set_max_batch_size") and hasattr(member, "__wrapped__"):
            setattr(instance, name, _unbatched(instance, member.__wrapped__))
    for route in app.routes:
        signature = getattr(getattr(route, "endpoint", None), "__signature__", None)
        if signature is not None and "self" in signature.parameters:
            dependency = signature.parameters["self"].default.dependency
            app.dependency_overrides[dependency] = lambda: instance
    try:
        with TestClient(app) as client:
            yield OffServe(instance, client)
    finally:
        app.dependency_overrides.clear()


def _unbatched(instance: Any, body: Any) -> Any:
    async def one(item: Any) -> Any:
        return (await body(instance, [item]))[0]

    return one


async def _awaited(awaitable: Any) -> Any:
    return await awaitable
