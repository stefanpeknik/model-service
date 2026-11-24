"""Make a marketplace's apps match the deployed analyses when the control plane starts.

It posts to the routes of the marketplace mock EMPAIA test installations run
(`/v1/custom-mock/portal-apps`, then the app's global configuration, such as
the case browser's viewer settings); a production marketplace lists apps through its
portal's review instead. App IDs derive from the EAD namespace, so registering
again replaces an app and every control-plane replica may do it. With a Medical Data
Service configured, the apps' preprocessing triggers follow (triggers.py).
Last, it deletes the apps it owns that no deployed analysis has any more, such
as one whose analysis was renamed: those with a `NAMESPACE_PREFIX` namespace
and an image on this deployment's JES target.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import httpx
import jwt

from model_service.providers.empaia.credentials import authorized_client
from model_service.providers.empaia.ead.apps import NAMESPACE_PREFIX


if TYPE_CHECKING:
    from collections.abc import Sequence

    from model_service.providers.empaia.credentials import OutgoingCredentials
    from model_service.providers.empaia.ead.apps import AnalysisApp
    from model_service.providers.empaia.triggers import PreprocessingTriggers


logger = logging.getLogger(__name__)

RETRY_SECONDS = 60.0
# Failures in a row after which they are logged as errors: by then it is a
# setting (URL, organization, client secret), not an outage.
ERROR_AFTER = 3


class Marketplace:
    """The EMPAIA marketplace the apps are registered in."""

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
        self._customer = f"{url.rstrip('/')}/v1/customer"
        self._organization_id = organization_id
        self._target = target
        self._credentials = credentials
        self._timeout = timeout
        self._transport = transport

    async def register(self, apps: Sequence[AnalysisApp]) -> None:
        """Post each app as a portal app, then its global configuration if it has one.

        Registering again replaces an app.

        Args:
            apps: The apps to register.

        Raises:
            httpx.HTTPError: If a request fails or is answered with an error status.
            jwt.InvalidTokenError: If the credentials' token cannot be decoded.
            KeyError: If a token response or discovery document lacks a field.
        """
        async with authorized_client(
            self._credentials, timeout=self._timeout, transport=self._transport
        ) as (client, headers):
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
                    "registered EMPAIA app %s (%s) of analysis %s",
                    app.app_id,
                    ", ".join(app.modes),
                    app.analysis_id,
                )

    async def retire_absent(self, apps: Sequence[AnalysisApp]) -> None:
        """Delete the owned portal apps that are not among the given apps.

        model-service owns a portal app when its namespace starts with
        `NAMESPACE_PREFIX` and its image points at this deployment's JES
        target, so RationAI's container apps under the same prefix and other
        deployments' apps are never touched.

        Args:
            apps: Every app that stays.

        Raises:
            httpx.HTTPError: If a request fails or is answered with an error status.
            jwt.InvalidTokenError: If the credentials' token cannot be decoded.
            KeyError: If the portal app list or a token response lacks a field.
        """
        keep = {app.portal_app_id for app in apps}
        async with authorized_client(
            self._credentials, timeout=self._timeout, transport=self._transport
        ) as (client, headers):
            # The mock ignores the query's filters and lists every portal app.
            response = await client.put(
                f"{self._customer}/portal-apps/query",
                json={},
                headers={**headers, "organization-id": self._organization_id},
            )
            response.raise_for_status()
            for portal_app in response.json()["items"]:
                namespace = self._owned_namespace(portal_app)
                if namespace is None or portal_app["id"] in keep:
                    continue
                deleted = await client.delete(
                    f"{self._mock}/portal-apps/{portal_app['id']}", headers=headers
                )
                # Another replica may have retired it first.
                if deleted.status_code == httpx.codes.NOT_FOUND:
                    continue
                deleted.raise_for_status()
                logger.info(
                    "retired EMPAIA portal app %s (%s): no deployed analysis has it",
                    portal_app["id"],
                    namespace,
                )

    def _owned_namespace(self, portal_app: dict[str, Any]) -> str | None:
        """Find the namespace of a portal app model-service registered here.

        Args:
            portal_app: A portal app as the marketplace lists it.

        Returns:
            The namespace of its first owned app view, or None if it isn't owned.

        Raises:
            KeyError: If an app view lacks its app, EAD or image.
        """
        image = f"http::{self._target}/"
        for view in portal_app["active_app_views"].values():
            if view is None:
                continue
            app = view["app"]
            namespace = str(app["ead"]["namespace"])
            if namespace.startswith(NAMESPACE_PREFIX) and str(
                app["registry_image_url"]
            ).startswith(image):
                return namespace
        return None


async def register_until_done(
    marketplace: Marketplace,
    apps: Sequence[AnalysisApp],
    triggers: PreprocessingTriggers | None = None,
    *,
    retry_seconds: float = RETRY_SECONDS,
) -> None:
    """Keep trying until every app and its triggers are registered; failures are logged, not raised.

    HTTP, token, `KeyError` and `ValueError` failures are retried; anything else
    propagates.

    Args:
        marketplace: The marketplace to register in.
        apps: The apps to register.
        triggers: The preprocessing triggers to align with the apps; None
            leaves triggers alone.
        retry_seconds: The wait between attempts.
    """
    failures = 0
    while True:
        try:
            await marketplace.register(apps)
            if triggers is not None:
                await triggers.ensure(apps)
            # Last, so owned triggers have left a retired app before it goes.
            await marketplace.retire_absent(apps)
            return
        except (httpx.HTTPError, jwt.InvalidTokenError, KeyError, ValueError) as error:
            failures += 1
            logger.log(
                logging.ERROR if failures >= ERROR_AFTER else logging.WARNING,
                "EMPAIA app registration failed %d time(s), retrying in %.0f s: %s",
                failures,
                retry_seconds,
                _reason(error),
            )
        await asyncio.sleep(retry_seconds)


def _reason(error: Exception) -> str:
    """Describe a registration failure without credentials.

    Args:
        error: The failure.

    Returns:
        The status and URL of a rejected request, else the error's type.
    """
    if isinstance(error, httpx.HTTPStatusError):
        # The URL carries no secret: credentials travel in headers.
        return f"HTTP {error.response.status_code} from {error.request.url}"
    return type(error).__name__
