"""Check AppService HTTP shapes, input materialization, and uncertain publication outcomes."""

from __future__ import annotations

import json
import unittest

import httpx
from aiohttp import web
from aiohttp.test_utils import TestServer
from rationai.empaia import Client

from model_service.providers.empaia.gateway import EmpaiaGatewayError
from model_service.providers.empaia.http_gateway import HttpEmpaiaGateway


OUTPUT_ID = "11111111-1111-4111-8111-111111111111"


class HttpEmpaiaGatewayTests(unittest.IsolatedAsyncioTestCase):
    """Exercise AppService HTTP shapes, input materialization, and uncertain publication outcomes."""

    async def appservice(self, transport: httpx.MockTransport) -> Client:
        """Serve HTTP response fixtures over loopback using the real SDK transport."""
        errors = []

        async def handle(request: web.Request) -> web.Response:
            external_request = httpx.Request(
                request.method,
                str(request.url),
                headers=request.headers,
                content=await request.read(),
            )
            try:
                response = await transport.handle_async_request(external_request)
            except httpx.TransportError:
                assert request.transport is not None
                request.transport.close()
                return web.Response()
            except Exception as error:
                errors.append(error)
                return web.Response(status=500)
            return web.Response(
                status=response.status_code,
                headers=response.headers,
                body=response.content,
            )

        self.addCleanup(self.assertEqual, errors, [])
        app = web.Application()
        app.router.add_route("*", "/{path:.*}", handle)
        server = TestServer(app)
        await server.start_server()
        self.addAsyncCleanup(server.close)
        client = Client(str(server.make_url("/")), "job-1", "credential")
        self.addAsyncCleanup(client.close)
        return client

    async def test_configuration_rejects_nested_nonfinite_values(self) -> None:
        """Verify configuration rejects nested nonfinite values."""
        gateway = HttpEmpaiaGateway(
            "https://empaia.example",
            "job-1",
            "credential",
            client=await self.appservice(
                httpx.MockTransport(
                    lambda _request: httpx.Response(
                        200,
                        content='{"global": {"nested": [NaN]}, "customer": {}}',
                    )
                )
            ),
        )
        try:
            with self.assertRaises(EmpaiaGatewayError):
                await gateway.get_configuration()
        finally:
            await gateway.close()

    async def test_mutation_guard_runs_before_authenticated_request(self) -> None:
        """Verify mutation guard runs before authenticated request."""
        events = []

        async def guard():
            """Record that mutation guarding ran before the fake authenticated HTTP request."""
            events.append("guard")

        def handle(request: httpx.Request) -> httpx.Response:
            """Return scenario-specific mock HTTP responses and record or assert requests for this gateway test."""
            self.assertEqual(request.headers["Authorization"], "Bearer credential")
            events.append("request")
            return httpx.Response(503)

        gateway = HttpEmpaiaGateway(
            "https://empaia.example",
            "job-1",
            "credential",
            client=await self.appservice(httpx.MockTransport(handle)),
        )
        gateway.set_before_mutation(guard)
        try:
            with self.assertRaises(EmpaiaGatewayError) as raised:
                await gateway.finalize()
            self.assertEqual(events, ["guard", "request"])
            self.assertTrue(raised.exception.retryable)
            self.assertTrue(raised.exception.outcome_unknown)
        finally:
            await gateway.close()

    async def test_configuration_is_flattened(self) -> None:
        """Verify configuration is flattened."""

        def handle(request: httpx.Request) -> httpx.Response:
            """Return scenario-specific mock HTTP responses and record or assert requests for this gateway test."""
            if request.url.path.endswith("/configuration"):
                return httpx.Response(
                    200,
                    json={"global": {"stride": 0.5}, "customer": {"tiles": 8}},
                )
            raise AssertionError(request.url.path)

        client = await self.appservice(httpx.MockTransport(handle))
        gateway = HttpEmpaiaGateway(
            "https://empaia.example",
            "job-1",
            "credential",
            client=client,
        )

        configuration = await gateway.get_configuration()

        self.assertEqual(configuration, {"stride": 0.5, "tiles": 8})
        await gateway.close()

    async def test_inline_outputs_and_lifecycle_use_appservice_contract(self) -> None:
        """Verify inline outputs and lifecycle use AppService contract."""
        requests: list[tuple[str, str, object | None]] = []

        def handle(request: httpx.Request) -> httpx.Response:
            """Return scenario-specific mock HTTP responses and record or assert requests for this gateway test."""
            body = json.loads(request.content) if request.content else None
            requests.append((request.method, request.url.path, body))
            if request.method == "POST":
                return httpx.Response(200, json={**body, "id": OUTPUT_ID})
            return httpx.Response(200, json={})

        gateway = HttpEmpaiaGateway(
            "https://empaia.example",
            "job-1",
            "credential",
            client=await self.appservice(httpx.MockTransport(handle)),
        )

        await gateway.post_primitive("score", "float", 0.75)
        await gateway.finalize()
        await gateway.report_failure("failed safely")
        await gateway.close()

        method, path, body = requests[0]
        self.assertEqual((method, path), ("POST", "/v3/job-1/outputs/score"))
        # The SDK sends the whole typed model; these are the fields that matter.
        self.assertEqual(
            {
                key: body[key]
                for key in ("name", "type", "value", "creator_id", "creator_type")
            },
            {
                "name": "score",
                "type": "float",
                "value": 0.75,
                "creator_id": "job-1",
                "creator_type": "job",
            },
        )
        self.assertEqual(
            requests[1:],
            [
                ("PUT", "/v3/job-1/finalize", None),
                ("PUT", "/v3/job-1/failure", {"user_message": "failed safely"}),
            ],
        )

    async def test_input_presence_comes_from_appservice_404(self) -> None:
        """Verify a 404 means the job lacks the input, while other failures stay errors."""

        def handle(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/inputs/slide"):
                return httpx.Response(200, json={"id": "slide-1"})
            if request.url.path.endswith("/inputs/region_of_interest"):
                return httpx.Response(404, json={"detail": "No such input"})
            return httpx.Response(503)

        gateway = HttpEmpaiaGateway(
            "https://empaia.example",
            "job-1",
            "credential",
            client=await self.appservice(httpx.MockTransport(handle)),
        )
        try:
            self.assertTrue(await gateway.has_input("slide"))
            self.assertFalse(await gateway.has_input("region_of_interest"))
            with self.assertRaises(EmpaiaGatewayError) as raised:
                await gateway.has_input("other")
            self.assertTrue(raised.exception.retryable)
        finally:
            await gateway.close()

    async def test_configuration_rejects_ambiguous_sections(self) -> None:
        """Verify configuration rejects ambiguous sections."""
        gateway = HttpEmpaiaGateway(
            "https://empaia.example",
            "job-1",
            "credential",
            client=await self.appservice(
                httpx.MockTransport(
                    lambda _request: httpx.Response(
                        200,
                        json={"global": {"tiles": 8}, "customer": {"tiles": 16}},
                    )
                )
            ),
        )

        with self.assertRaises(EmpaiaGatewayError) as raised:
            await gateway.get_configuration()

        self.assertFalse(raised.exception.retryable)
        await gateway.close()


if __name__ == "__main__":
    unittest.main()
