"""Credentials model-service presents to EMPAIA services it calls on its own behalf.

Job calls to the App Service use the job's token instead. `ClientCredentials`
does what `empaia-sender-auth` does: it finds the identity provider's token
endpoint through OpenID discovery, requests a client-credentials token and
renews it after half its lifetime.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jwt


if TYPE_CHECKING:
    import httpx


class OutgoingCredentials(ABC):
    @abstractmethod
    async def authorization(self, client: httpx.AsyncClient) -> str | None:
        """Return the `Authorization` header value, or None to send none.

        Args:
            client: The HTTP client to fetch a token with.

        Returns:
            The header value, or None.
        """
        ...


class NoCredentials(OutgoingCredentials):
    async def authorization(self, client: httpx.AsyncClient) -> str | None:
        """Send no `Authorization` header.

        Args:
            client: Unused.

        Returns:
            None.
        """
        return None


class ClientCredentials(OutgoingCredentials):
    def __init__(self, idp_url: str, client_id: str, client_secret: str) -> None:
        """Keep the client's credentials; the first call fetches a token.

        Args:
            idp_url: The identity provider's base URL; discovery is below it at
                `/.well-known/openid-configuration`.
            client_id: The OAuth client ID.
            client_secret: The OAuth client secret.
        """
        self._discovery_url = f"{idp_url.rstrip('/')}/.well-known/openid-configuration"
        self._client_id = client_id
        self._client_secret = client_secret
        self._token: str | None = None
        self._renew_at = 0.0
        self._lock = asyncio.Lock()

    async def authorization(self, client: httpx.AsyncClient) -> str | None:
        """Return a bearer token, renewing it once half its lifetime has passed.

        Concurrent callers share one renewal.

        Args:
            client: The HTTP client to fetch a token with.

        Returns:
            `Bearer <token>`.

        Raises:
            httpx.HTTPError: If discovery or the token request fails.
            KeyError: If the discovery document or token response lacks a field, or
                the token has no `exp` claim.
            jwt.InvalidTokenError: If the token cannot be decoded.
        """
        async with self._lock:
            if self._token is None or time.time() >= self._renew_at:
                await self._renew(client)
            return f"Bearer {self._token}"

    async def _renew(self, client: httpx.AsyncClient) -> None:
        """Fetch a client-credentials token from the endpoint OpenID discovery names.

        The token's claims are read without verifying its signature, only to renew it
        after half its lifetime.

        Args:
            client: The HTTP client to fetch a token with.

        Raises:
            httpx.HTTPError: If discovery or the token request fails.
            KeyError: If the discovery document or token response lacks a field, or
                the token has no `exp` claim.
            jwt.InvalidTokenError: If the token cannot be decoded.
        """
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
