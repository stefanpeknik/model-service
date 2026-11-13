"""Check how the HTTP gateway classifies App Service failures, through loopback requests."""

from __future__ import annotations

import unittest

from aiohttp import web
from aiohttp.test_utils import TestServer
from rationai.empaia import Client

from model_service.providers.empaia.app_service.gateway import (
    EmpaiaGatewayError,
)
from model_service.providers.empaia.app_service.http import (
    HttpEmpaiaGateway,
)


class SdkGatewayContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_after_accepted_put_is_unknown_without_resend(self):
        requests = []

        async def handle(request):
            requests.append(request.method)
            if request.method == "GET":
                return web.json_response({})
            if requests.count("PUT") == 1:
                request.transport.close()
            return web.Response(status=204)

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", handle)
        async with TestServer(app) as server:
            gateway = HttpEmpaiaGateway(str(server.make_url("/")), "job", "token")
            try:
                await gateway.get_configuration()
                with self.assertRaises(EmpaiaGatewayError) as raised:
                    await gateway.finalize()
                self.assertTrue(raised.exception.retryable)
                self.assertTrue(raised.exception.outcome_unknown)
            finally:
                await gateway.close()
        self.assertEqual(requests, ["GET", "PUT"])

    async def test_malformed_mutation_acknowledgment_is_outcome_unknown(self):
        received = []

        async def handle(request):
            received.append(request.path)
            return web.Response(text="private malformed response")

        app = web.Application()
        app.router.add_post("/v3/job/outputs/score", handle)
        async with TestServer(app) as server:
            gateway = HttpEmpaiaGateway(
                str(server.make_url("/")),
                "job",
                "private-token",
                client=Client(str(server.make_url("/")), "job", "private-token"),
            )
            try:
                with self.assertRaises(EmpaiaGatewayError) as raised:
                    await gateway.post_primitive("score", "float", 0.75)
                self.assertTrue(raised.exception.outcome_unknown)
                self.assertFalse(raised.exception.retryable)
                self.assertNotIn("private", raised.exception.safe_message)
            finally:
                await gateway.close()
        self.assertEqual(received, ["/v3/job/outputs/score"])

    async def test_an_unusable_acknowledgment_of_an_output_is_outcome_unknown(self):
        for value in ({}, {"id": ""}, {"id": 4}, None):
            with self.subTest(value=value):

                async def handle(_request, value=value):
                    return web.json_response(value)

                app = web.Application()
                app.router.add_post("/v3/job/outputs/score", handle)
                async with TestServer(app) as server:
                    gateway = HttpEmpaiaGateway(
                        str(server.make_url("/")), "job", "token"
                    )
                    self.addAsyncCleanup(gateway.close)
                    with self.assertRaises(EmpaiaGatewayError) as raised:
                        await gateway.post_primitive("score", "float", 0.75)
                    self.assertTrue(raised.exception.outcome_unknown)
                    self.assertFalse(raised.exception.retryable)
                    await gateway.close()

    async def test_http_errors_preserve_read_and_mutation_classification(self):
        for status in (400, 401, 403, 404, 429, 500, 503):
            with self.subTest(status=status):
                received = []

                async def handle(request, received=received, status=status):
                    received.append(request.method)
                    return web.Response(
                        status=status, text="private-token upstream detail"
                    )

                app = web.Application()
                app.router.add_route("*", "/{path:.*}", handle)
                async with TestServer(app) as server:
                    gateway = HttpEmpaiaGateway(
                        str(server.make_url("/")), "job", "token"
                    )
                    self.addAsyncCleanup(gateway.close)
                    for operation, mutation in (
                        (gateway.get_configuration, False),
                        (gateway.finalize, True),
                    ):
                        with self.assertRaises(EmpaiaGatewayError) as raised:
                            await operation()
                        transient = status == 429 or status >= 500
                        self.assertEqual(raised.exception.retryable, transient)
                        self.assertEqual(
                            raised.exception.outcome_unknown,
                            mutation and status >= 500,
                        )
                        self.assertNotIn("private", str(raised.exception))
                    await gateway.close()
                self.assertEqual(received, ["GET", "PUT"])
