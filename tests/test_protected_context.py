from __future__ import annotations

import base64
import unittest

from execution_scheduler.security import ProtectedContextCipher, ProtectedContextError


class ProtectedContextCipherTests(unittest.TestCase):
    def test_context_is_encrypted_and_authenticated(self) -> None:
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
