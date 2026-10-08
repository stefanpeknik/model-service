"""Check provider-context encryption and tamper detection."""

from __future__ import annotations

import base64
import unittest

from execution_scheduler.persistence.protected_context import (
    ProtectedContextCipher,
    ProtectedContextError,
)


class ProtectedContextCipherTests(unittest.TestCase):
    """Exercise provider-context encryption and tamper detection."""

    def test_context_is_encrypted_and_authenticated(self) -> None:
        """Verify context is encrypted and authenticated.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        key = bytes(range(32))
        cipher = ProtectedContextCipher.from_base64(
            base64.urlsafe_b64encode(key).decode()
        )
        context = {
            "credential": "secret-job-token",
            "installation": "test-empaia",
        }

        encrypted = cipher.encrypt(context)

        self.assertNotIn(b"secret-job-token", encrypted)
        self.assertEqual(cipher.decrypt(encrypted), context)
        tampered = encrypted[:-1] + bytes([encrypted[-1] ^ 1])
        with self.assertRaisesRegex(
            ProtectedContextError,
            "protected provider context failed authentication",
        ):
            cipher.decrypt(tampered)


if __name__ == "__main__":
    unittest.main()
