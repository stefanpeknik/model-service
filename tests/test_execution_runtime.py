from __future__ import annotations

import base64
import unittest
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient, MockTransport, Request, Response

from execution_scheduler.runtime import create_app
from execution_scheduler.settings import SchedulerSettings


class ExecutionRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_public_inference_uses_separate_origin_without_changing_workflow_health(
        self,
    ) -> None:
        forwarded: list[Request] = []

        def serve_handler(request: Request) -> Response:
            forwarded.append(request)
            if request.url == "http://workflow-serve/-/routes":
                return Response(
                    200, json={"/workflows/prostate-heatmap": "prostate-heatmap"}
                )
            if request.url == "http://model-serve/prostate-classifier-1":
                return Response(200, json=0.25)
            return Response(404)

        settings = SchedulerSettings.from_environment(
            {
                "DATABASE_URL": "postgresql://unused",
                "PROVIDER_CONTEXT_KEY": base64.urlsafe_b64encode(b"x" * 32).decode(),
                "RAY_SERVE_ORIGIN": "http://workflow-serve",
                "INFERENCE_SERVE_ORIGIN": "http://model-serve",
            }
        )
        async with AsyncClient(transport=MockTransport(serve_handler)) as serve_client:
            with patch(
                "execution_scheduler.runtime.httpx.AsyncClient",
                return_value=serve_client,
            ):
                app = create_app(settings)
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://model-service",
            ) as client:
                workflows = await client.get("/healthz/workflows")
                prediction = await client.post(
                    "/prostate-classifier-1", content=b"compressed-image"
                )

        self.assertEqual(workflows.status_code, 200)
        self.assertEqual(prediction.status_code, 200)
        self.assertEqual(prediction.json(), 0.25)
        self.assertEqual(forwarded[0].url, "http://workflow-serve/-/routes")
        self.assertEqual(forwarded[1].url, "http://model-serve/prostate-classifier-1")
        self.assertEqual(forwarded[1].content, b"compressed-image")
        self.assertEqual(forwarded[1].extensions["timeout"]["read"], 900)


if __name__ == "__main__":
    unittest.main()
