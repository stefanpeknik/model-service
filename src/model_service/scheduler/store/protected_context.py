"""Encrypt private provider context at rest using a versioned AES-GCM envelope.

The envelope carries a fresh nonce and authenticates a fixed context identifier. Public
invocations and execution views must not contain these credentials.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
from typing import TYPE_CHECKING, cast

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.builders.contracts import JsonValue


_VERSION = b"\x01"
_ASSOCIATED_DATA = b"model-service/provider-context/v1"
_NONCE_BYTES = 12


class ProtectedContextError(ValueError):
    """Signal failed authentication of stored encrypted provider context."""


class ProtectedContextCipher:
    """Encrypt and authenticate provider context using one configured 256-bit key."""

    def __init__(self, key: bytes) -> None:
        """Require a 32-byte AES key and construct its authenticated encryption primitive.

        Args:
            key: Raw AES-256 key.

        Raises:
            ValueError: The key is not 32 bytes long.
        """
        if len(key) != 32:
            raise ValueError("provider-context encryption key must contain 32 bytes")
        self._cipher = AESGCM(key)

    @classmethod
    def from_base64(cls, value: str) -> ProtectedContextCipher:
        """Decode a validated URL-safe base64 key and require the decoded AES-256 key length.

        Args:
            value: URL-safe base64 encoding of the key.

        Returns:
            A cipher using the decoded key.

        Raises:
            ValueError: The value is not URL-safe base64 or does not decode to 32 bytes.
        """
        try:
            key = base64.b64decode(value, altchars=b"-_", validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError(
                "provider-context encryption key must be URL-safe base64"
            ) from error
        return cls(key)

    def encrypt(self, context: Mapping[str, JsonValue]) -> bytes:
        """Encode finite JSON and return a version byte, random nonce, and authenticated ciphertext.

        Args:
            context: Provider context to protect.

        Returns:
            The encrypted envelope.

        Raises:
            ValueError: The context contains NaN or an infinite number.
        """
        plaintext = json.dumps(
            context,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        nonce = os.urandom(_NONCE_BYTES)
        ciphertext = self._cipher.encrypt(nonce, plaintext, _ASSOCIATED_DATA)
        return _VERSION + nonce + ciphertext

    def decrypt(self, value: bytes) -> dict[str, JsonValue]:
        """Validate the envelope, authenticate and decrypt it, then require a JSON object.

        Args:
            value: Encrypted envelope from encrypt.

        Returns:
            The decrypted provider context.

        Raises:
            ValueError: The envelope version or length is unsupported, or the JSON holds NaN or Infinity.
            ProtectedContextError: Authentication failed, as with corrupt ciphertext or a wrong key.
            TypeError: The decrypted JSON is not an object.
        """
        if len(value) <= len(_VERSION) + _NONCE_BYTES or not value.startswith(_VERSION):
            raise ValueError("protected provider context has an unsupported envelope")
        nonce_start = len(_VERSION)
        ciphertext_start = nonce_start + _NONCE_BYTES
        try:
            plaintext = self._cipher.decrypt(
                value[nonce_start:ciphertext_start],
                value[ciphertext_start:],
                _ASSOCIATED_DATA,
            )
        except InvalidTag as error:
            raise ProtectedContextError(
                "protected provider context failed authentication"
            ) from error
        decoded = json.loads(
            plaintext,
            parse_constant=lambda value: _reject_constant(value),
        )
        if not isinstance(decoded, dict):
            raise TypeError("protected provider context must contain an object")
        return cast("dict[str, JsonValue]", decoded)


def _reject_constant(value: str) -> None:
    """Reject nonstandard JSON constants such as NaN or Infinity in decrypted context.

    Args:
        value: The constant's name as it appears in the JSON.

    Raises:
        ValueError: Always.
    """
    raise ValueError(f"invalid JSON constant {value}")
