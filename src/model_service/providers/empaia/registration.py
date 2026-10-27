"""Register the deployed models' EMPAIA apps in a marketplace when the facade starts.

It posts to the route of the marketplace mock EMPAIA test installations run
(`/v1/custom-mock/portal-apps`); a production marketplace lists apps through its
portal's review instead. App IDs derive from the EAD namespace, so registering
again replaces an app and every facade replica may do it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import httpx


if TYPE_CHECKING:
    from collections.abc import Sequence

    from model_service.providers.empaia.apps import AppDefinition
    from model_service.providers.empaia.credentials import OutgoingCredentials


logger = logging.getLogger(__name__)

RETRY_SECONDS = 60.0


class Marketplace:
    def __init__(
        self,
        url: str,
        organization_id: str,
        target: str,
        credentials: OutgoingCredentials,
        *,
        timeout: float = 30,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = f"{url.rstrip('/')}/v1/custom-mock/portal-apps"
        self._organization_id = organization_id
        self._target = target
        self._credentials = credentials
        self._timeout = timeout
        self._transport = transport

    async def register(self, apps: Sequence[AppDefinition]) -> None:
        async with httpx.AsyncClient(
            timeout=self._timeout, transport=self._transport
        ) as client:
            headers = {}
            if (
                authorization := await self._credentials.authorization(client)
            ) is not None:
                headers["Authorization"] = authorization
            for app in apps:
                response = await client.post(
                    self._url,
                    json=app.portal_app(self._target, self._organization_id),
                    headers=headers,
                )
                response.raise_for_status()
                logger.info(
                    "registered EMPAIA app %s (%s, %s on %s)",
                    app.app_id,
                    app.mode,
                    app.builder_id,
                    app.model_id,
                )


async def register_until_done(
    marketplace: Marketplace,
    apps: Sequence[AppDefinition],
    *,
    retry_seconds: float = RETRY_SECONDS,
) -> None:
    """Keep trying until every app is registered; failures are logged, not raised."""
    while True:
        try:
            await marketplace.register(apps)
            return
        except (httpx.HTTPError, KeyError, ValueError) as error:
            logger.warning(
                "EMPAIA app registration failed, retrying in %.0f s: %s",
                retry_seconds,
                type(error).__name__,
            )
        await asyncio.sleep(retry_seconds)
