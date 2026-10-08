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

    from workflow_execution.contracts import JsonValue


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
            key: Exactly 32 bytes of AES-256 key material.

        Raises:
            ValueError: key does not contain exactly 32 bytes.
        """
        if len(key) != 32:
            raise ValueError("provider-context encryption key must contain 32 bytes")
        self._cipher = AESGCM(key)

    @classmethod
    def from_base64(cls, value: str) -> ProtectedContextCipher:
        """Decode a validated URL-safe base64 key and require the decoded AES-256 key length.

        Args:
            value: URL-safe base64 text encoding exactly 32 bytes of AES key material.

        Returns:
            ProtectedContextCipher: Configured authenticated provider-context cipher.

        Raises:
            ValueError: value is not valid URL-safe base64 or does not decode to exactly
                32 bytes.
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
            context: Private JSON provider state to serialize and encrypt.

        Returns:
            bytes: Version byte followed by a fresh 12-byte nonce and authenticated
                ciphertext for the finite JSON context.

        Raises:
            TypeError: context contains an object that JSON cannot serialize.
            ValueError: context contains non-finite numbers or circular references.
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
        """Validate the envelope, authenticate and decrypt it, then require a JSON object. Corrupt ciphertext raises ProtectedContextError.

        Args:
            value: Stored versioned AES-GCM envelope containing nonce and ciphertext.

        Returns:
            dict[str, JsonValue]: Authenticated private provider JSON object.

        Raises:
            ProtectedContextError: Ciphertext authentication fails with the configured
                key and associated data.
            ValueError: The envelope version/length is unsupported or decrypted data
                contains an invalid JSON constant.
            json.JSONDecodeError: Authenticated plaintext is not valid JSON.
            TypeError: Authenticated JSON is not an object.
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
        value: Nonstandard JSON constant encountered while decoding decrypted provider
            context.

    Returns:
        None: No return value.

    Raises:
        ValueError: JSON contains a non-finite constant such as NaN or Infinity.
    """
    raise ValueError(f"invalid JSON constant {value}")
