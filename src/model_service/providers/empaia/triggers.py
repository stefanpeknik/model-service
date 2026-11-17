"""Keep the installation's preprocessing triggers in line with the registered apps.

A trigger names a portal app and a tissue and stain; EMPAIA's Workbench Daemon
starts a preprocessing job of that app for every new slide tagged with them.
Triggers live in the Medical Data Service (`/v3/preprocessing-triggers`), which
model-service calls with its own client credentials. Those it creates carry
its owner ID as `creator_id`: it creates the ones the model files list that
no trigger covers yet, deletes its own that they no longer list, and never
touches triggers someone else created.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx


if TYPE_CHECKING:
    from collections.abc import Sequence

    from model_service.providers.empaia.credentials import OutgoingCredentials
    from model_service.providers.empaia.ead.apps import AppDefinition


logger = logging.getLogger(__name__)

type TriggerKey = tuple[str, str, str]


class PreprocessingTriggers:
    """The Medical Data Service's preprocessing triggers, as far as model-service owns them."""

    def __init__(
        self,
        url: str,
        owner: str,
        credentials: OutgoingCredentials,
        *,
        timeout: float = 30,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Keep the Medical Data Service connection settings.

        Args:
            url: The Medical Data Service's base URL, without `/v3`.
            owner: The `creator_id` that marks triggers as model-service's.
            credentials: Supplies the `Authorization` header.
            timeout: The HTTP timeout, in seconds.
            transport: An httpx transport, e.g. for tests; the default if None.
        """
        self._triggers = f"{url.rstrip('/')}/v3/preprocessing-triggers"
        self._owner = owner
        self._credentials = credentials
        self._timeout = timeout
        self._transport = transport

    async def ensure(self, apps: Sequence[AppDefinition]) -> None:
        """Create the apps' missing triggers and delete owned ones they no longer list.

        A listed pair counts as present when someone else's trigger already
        names it, so the same pair never starts two jobs per slide. Of owned
        duplicates, the lowest ID stays. After creating triggers it looks once
        more, so API replicas that create the same trigger at once end with one.

        Args:
            apps: Every registered app; those without triggers want none.

        Raises:
            httpx.HTTPError: If a request fails or is answered with an error status.
            jwt.InvalidTokenError: If the credentials' token cannot be decoded.
            KeyError: If the trigger list or a token response lacks a field.
        """
        wanted: set[TriggerKey] = {
            (app.portal_app_id, trigger.tissue, trigger.stain)
            for app in apps
            for trigger in app.triggers
        }
        async with httpx.AsyncClient(
            timeout=self._timeout, transport=self._transport
        ) as client:
            headers = {}
            if (
                authorization := await self._credentials.authorization(client)
            ) is not None:
                headers["Authorization"] = authorization
            if await self._align(client, headers, wanted):
                await self._align(client, headers, wanted)

    async def _align(
        self, client: httpx.AsyncClient, headers: dict[str, str], wanted: set[TriggerKey]
    ) -> bool:
        """Delete owned triggers that are unwanted or duplicate, then create missing ones.

        Args:
            client: The HTTP client.
            headers: The `Authorization` header, if any.
            wanted: The pairs the apps list, with their portal app IDs.

        Returns:
            Whether it created any trigger.

        Raises:
            httpx.HTTPError: If a request fails or is answered with an error status.
            KeyError: If the trigger list lacks a field.
        """
        response = await client.get(self._triggers, headers=headers)
        response.raise_for_status()
        items = response.json()["items"]
        owned = sorted(
            (item for item in items if item["creator_id"] == self._owner),
            key=lambda item: item["id"],
        )
        present = {
            (item["portal_app_id"], item["tissue"], item["stain"])
            for item in items
            if item["creator_id"] != self._owner
        }
        for item in owned:
            key = (item["portal_app_id"], item["tissue"], item["stain"])
            if key in wanted and key not in present:
                present.add(key)
                continue
            deleted = await client.delete(
                f"{self._triggers}/{item['id']}", headers=headers
            )
            # Another replica may have deleted the same duplicate.
            if deleted.status_code != httpx.codes.NOT_FOUND:
                deleted.raise_for_status()
            logger.info("deleted preprocessing trigger %s", item["id"])
        missing = sorted(wanted - present)
        for portal_app_id, tissue, stain in missing:
            created = await client.post(
                self._triggers,
                json={
                    "creator_id": self._owner,
                    # The only creator type the Medical Data Service knows.
                    "creator_type": "USER",
                    "portal_app_id": portal_app_id,
                    "tissue": tissue,
                    "stain": stain,
                },
                headers=headers,
            )
            created.raise_for_status()
            logger.info(
                "created preprocessing trigger for portal app %s on %s %s",
                portal_app_id,
                tissue,
                stain,
            )
        return bool(missing)
