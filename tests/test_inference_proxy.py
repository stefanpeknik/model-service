"""Check binary model request forwarding and permitted response headers."""

from __future__ import annotations

import unittest

import httpx

from model_service.inference import HttpInferenceProxy, InferenceRequest


class HttpInferenceProxyTests(unittest.IsolatedAsyncioTestCase):
    """Exercise binary model request forwarding and permitted response headers."""

    async def test_forwards_legacy_request_and_binary_response(self) -> None:
        """Verify forwards legacy request and binary response."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            """Assert the forwarded binary request and return an upstream response with inference metadata."""
            captured.append(request)
            return httpx.Response(
                200,
                content=b"compressed-result",
                headers={
                    "content-type": "application/octet-stream",
                    "x-output-shape": "(1, 2, 2)",
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            proxy = HttpInferenceProxy(client, origin="http://ray-serve")
            response = await proxy.forward(
                InferenceRequest(
                    path="classifier",
                    query="variant=probe",
                    body=b"compressed-image",
                    headers={
                        "content-type": "application/octet-stream",
                        "x-output-dtype": "float32",
                    },
                )
            )

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].url, "http://ray-serve/classifier?variant=probe")
        self.assertEqual(captured[0].content, b"compressed-image")
        self.assertEqual(captured[0].headers["x-output-dtype"], "float32")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"compressed-result")
        self.assertEqual(response.headers["x-output-shape"], "(1, 2, 2)")
        self.assertNotIn("content-length", response.headers)


if __name__ == "__main__":
    unittest.main()
