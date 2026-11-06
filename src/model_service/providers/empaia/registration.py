"""Register the deployed models' EMPAIA apps in a marketplace when the facade starts.

It posts to the routes of the marketplace mock EMPAIA test installations run
(`/v1/custom-mock/portal-apps`, then the app's global configuration, such as
the case browser's viewer settings); a production marketplace lists apps through its
portal's review instead. App IDs derive from the EAD namespace, so registering
again replaces an app and every facade replica may do it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import httpx
import jwt


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
        """Keep the marketplace connection settings.

        Args:
            url: The marketplace's base URL; the mock's routes are below
                `/v1/custom-mock`.
            organization_id: The organization that owns the apps.
            target: model-service's name in JES's HTTP targets, part of each image.
            credentials: Supplies the `Authorization` header.
            timeout: The HTTP timeout, in seconds.
            transport: An httpx transport, e.g. for tests; the default if None.
        """
        self._mock = f"{url.rstrip('/')}/v1/custom-mock"
        self._organization_id = organization_id
        self._target = target
        self._credentials = credentials
        self._timeout = timeout
        self._transport = transport

    async def register(self, apps: Sequence[AppDefinition]) -> None:
        """Post each app as a portal app, then its global configuration if it has one.

        Registering again replaces an app.

        Args:
            apps: The apps to register.

        Raises:
            httpx.HTTPError: If a request fails or is answered with an error status.
            jwt.InvalidTokenError: If the credentials' token cannot be decoded.
            KeyError: If a token response or discovery document lacks a field.
        """
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
                    f"{self._mock}/portal-apps",
                    json=app.portal_app(self._target, self._organization_id),
                    headers=headers,
                )
                response.raise_for_status()
                if app.global_configuration:
                    response = await client.put(
                        f"{self._mock}/apps/{app.app_id}/config/global",
                        json=app.global_configuration,
                        headers=headers,
                    )
                    response.raise_for_status()
                logger.info(
                    "registered EMPAIA app %s (%s, %s on %s)",
                    app.app_id,
                    ", ".join(app.modes),
                    app.builder_id,
                    app.model_id,
                )


async def register_until_done(
    marketplace: Marketplace,
    apps: Sequence[AppDefinition],
    *,
    retry_seconds: float = RETRY_SECONDS,
) -> None:
    """Keep trying until every app is registered; failures are logged, not raised.

    HTTP, token, `KeyError` and `ValueError` failures are retried; anything else
    propagates.

    Args:
        marketplace: The marketplace to register in.
        apps: The apps to register.
        retry_seconds: The wait between attempts.
    """
    while True:
        try:
            await marketplace.register(apps)
            return
        except (httpx.HTTPError, jwt.InvalidTokenError, KeyError, ValueError) as error:
            logger.warning(
                "EMPAIA app registration failed, retrying in %.0f s: %s",
                retry_seconds,
                type(error).__name__,
            )
        await asyncio.sleep(retry_seconds)
