"""Check expiry-only credential inspection through EMPAIA preparation."""

from __future__ import annotations

import base64
import json
import unittest
from datetime import UTC, datetime, timedelta

import jwt

from execution_scheduler.contracts import InvalidScheduleRequestError, ScheduleRequest
from integrations.empaia.provider import EmpaiaProviderAdapter
from tests.test_empaia_execution import ControlledEmpaiaService


API_URL = "https://empaia.invalid"


class EmpaiaCredentialTests(unittest.IsolatedAsyncioTestCase):
    """Validate token structure and deadline caps without authenticating JWT signatures."""

    def setUp(self) -> None:
        """Create a provider with a controlled gateway and a future execution deadline."""
        self.service = ControlledEmpaiaService()
        self.provider = EmpaiaProviderAdapter(
            self.service.gateway, allowed_api_urls=(API_URL,)
        )
        self.deadline = datetime.now(UTC) + timedelta(hours=1)

    async def test_integer_and_fractional_expiry_preserve_utc_deadline_caps(
        self,
    ) -> None:
        """Keep NumericDate seconds and fractional seconds in the prepared execution."""
        for expiry in (
            int(self.deadline.timestamp()) - 300,
            self.deadline.timestamp() - 300.125,
        ):
            with self.subTest(expiry=expiry):
                prepared = await self.provider.prepare(
                    self._request(_token({"exp": expiry})), self.deadline
                )
                expected = datetime.fromtimestamp(expiry, tz=UTC)
                self.assertEqual(prepared.deadline_cap, expected)
                self.assertEqual(prepared.terminal_report_deadline_at, expected)
                self.assertEqual(prepared.invocation.deadline_at, expected)

    async def test_later_expiry_does_not_extend_the_execution_deadline(self) -> None:
        """Use the earlier execution deadline even when the credential lasts longer."""
        expiry = self.deadline + timedelta(hours=1)
        prepared = await self.provider.prepare(
            self._request(_token({"exp": expiry.timestamp()})), self.deadline
        )
        self.assertEqual(prepared.deadline_cap, expiry)
        self.assertEqual(prepared.invocation.deadline_at, self.deadline)

    async def test_inspection_does_not_authenticate_signatures_or_other_claims(
        self,
    ) -> None:
        """Read only expiry; authentication and other claim policy are separate concerns."""
        token = jwt.encode(
            {
                "exp": self.deadline.timestamp(),
                "nbf": self.deadline.timestamp() + 60,
                "aud": "not-model-service",
            },
            "synthetic-test-key-not-trusted-0001",
            algorithm="HS256",
        )
        prepared = await self.provider.prepare(self._request(token), self.deadline)
        self.assertEqual(prepared.deadline_cap, self.deadline)
        self.assertEqual(prepared.protected_context["credential"], token)

    async def test_expired_claim_is_returned_for_existing_admission_deadline_policy(
        self,
    ) -> None:
        """Let the scheduler enforce expiry rather than adding a second clock policy."""
        expiry = datetime.now(UTC) - timedelta(minutes=1)
        prepared = await self.provider.prepare(
            self._request(_token({"exp": expiry.timestamp()})), self.deadline
        )
        self.assertEqual(prepared.deadline_cap, expiry)

    async def test_invalid_expiry_is_rejected_before_gateway_preparation(self) -> None:
        """Reject coerced, nonfinite, and unrepresentable expiry with a safe error."""
        for expiry in (
            None,
            True,
            False,
            "123",
            [],
            {},
            float("nan"),
            float("inf"),
            float("-inf"),
            1e300,
            -1e300,
            1e20,
            10**400,
        ):
            with self.subTest(expiry=expiry):
                await self._assert_invalid(_token({"exp": expiry}))
        self.assertEqual(self.service.configuration_reads, 0)
        self.assertEqual(self.service.credentials, [])

    async def test_missing_claim_and_malformed_token_are_safe_admission_errors(
        self,
    ) -> None:
        """Translate JWT decoding errors without exposing the original credential."""
        for token in (
            "not-a-token",
            "a.b.c",
            "e30.!!!.",
            _token({}),
            _token([]),
            _token({"exp": 123}, header=[]),
        ):
            with self.subTest(token_shape=token.count(".")):
                await self._assert_invalid(token)
        self.assertEqual(self.service.configuration_reads, 0)

    async def test_nested_malformed_payload_raises_a_safe_error(self) -> None:
        """Cover the pre-verification payload parser regression fixed in PyJWT 2.15."""
        payload = (
            base64.urlsafe_b64encode(b"[" * 2000 + b"]" * 2000).rstrip(b"=").decode()
        )
        await self._assert_invalid(f"e30.{payload}.")

    async def _assert_invalid(self, token: str) -> None:
        """Assert the provider returns a safe validation error for the supplied token."""
        with self.assertRaises(InvalidScheduleRequestError) as raised:
            await self.provider.prepare(self._request(token), self.deadline)
        self.assertEqual(
            str(raised.exception), "EMPAIA credential does not contain a valid expiry"
        )
        self.assertNotIn(token, str(raised.exception))

    @staticmethod
    def _request(token: str) -> ScheduleRequest:
        """Build a synthetic job request without contacting an external AppService."""
        return ScheduleRequest(
            builder_id="heatmap",
            model_id="prostate-classifier-1",
            provider="empaia",
            body={
                "installation_id": "test-installation",
                "api_url": "https://empaia.invalid",
                "job_id": "test-job",
                "credential": token,
            },
        )


def _token(payload: object, *, header: object | None = None) -> str:
    """Encode synthetic token sections with an empty unverified signature."""

    def encode(value: object) -> str:
        """Encode fixture JSON using JWT's unpadded Base64url representation."""
        return (
            base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()
        )

    return f"{encode({} if header is None else header)}.{encode(payload)}."


if __name__ == "__main__":
    unittest.main()
