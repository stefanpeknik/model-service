"""Credentials model-service presents to EMPAIA services it calls on its own behalf.

Job calls to the App Service use the job's token instead. `ClientCredentials`
does what `empaia-sender-auth` does: it finds the identity provider's token
endpoint through OpenID discovery, requests a client-credentials token and
renews it after half its lifetime.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Protocol

import jwt


if TYPE_CHECKING:
    import httpx


class OutgoingCredentials(Protocol):
    async def authorization(self, client: httpx.AsyncClient) -> str | None:
        """The `Authorization` header value, or None to send none."""
        ...


class NoCredentials:
    async def authorization(self, client: httpx.AsyncClient) -> str | None:
        return None


class ClientCredentials:
    def __init__(self, idp_url: str, client_id: str, client_secret: str) -> None:
        self._discovery_url = f"{idp_url.rstrip('/')}/.well-known/openid-configuration"
        self._client_id = client_id
        self._client_secret = client_secret
        self._token: str | None = None
        self._renew_at = 0.0
        self._lock = asyncio.Lock()

    async def authorization(self, client: httpx.AsyncClient) -> str | None:
        async with self._lock:
            if self._token is None or time.time() >= self._renew_at:
                await self._renew(client)
            return f"Bearer {self._token}"

    async def _renew(self, client: httpx.AsyncClient) -> None:
        discovery = await client.get(self._discovery_url)
        discovery.raise_for_status()
        response = await client.post(
            discovery.json()["token_endpoint"],
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
        )
        response.raise_for_status()
        token = response.json()["access_token"]
        claims = jwt.decode(token, options={"verify_signature": False})
        issued = float(claims.get("iat", time.time()))
        self._token = token
        self._renew_at = issued + (float(claims["exp"]) - issued) / 2
