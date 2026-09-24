from __future__ import annotations

import os
import unittest
from datetime import timedelta


os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


@unittest.skipUnless(
    os.environ.get("RUN_RAY_SERVE_INTEGRATION") == "1",
    "set RUN_RAY_SERVE_INTEGRATION=1 to run local Ray Serve validation",
)
class ValidationRawInferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_classifies_through_public_scheduler_and_ray_serve(self) -> None:
        import httpx
        import numpy as np
        import ray
        from rationai import AsyncClient
        from ray import serve

        from execution_scheduler.api import create_execution_app
        from execution_scheduler.inference import HttpInferenceProxy
        from execution_scheduler.repository import InMemoryExecutionRepository
        from execution_scheduler.scheduler import ExecutionScheduler
        from execution_scheduler.standalone import StandaloneProviderAdapter
        from validation.classifier_serve import ValidationProstateClassifier
        from workflow_execution.catalog import WorkflowCatalog

        catalog = WorkflowCatalog.default()
        provider = StandaloneProviderAdapter(catalog)
        scheduler = ExecutionScheduler(
            repository=InMemoryExecutionRepository(),
            providers={provider.name: provider},
            known_workflows=catalog.identifiers,
            default_timeout=timedelta(minutes=5),
            maximum_timeout=timedelta(minutes=5),
        )
        ray.init(num_cpus=2, include_dashboard=False)
        try:
            serve.run(
                ValidationProstateClassifier.options(  # type: ignore[attr-defined]
                    ray_actor_options={"num_cpus": 0}
                ).bind(),
                name="prostate-classifier-1",
                route_prefix="/prostate-classifier-1",
            )
            async with httpx.AsyncClient() as ray_client:
                app = create_execution_app(
                    scheduler,
                    inference_proxy=HttpInferenceProxy(
                        ray_client, origin="http://127.0.0.1:8000"
                    ),
                )
                async with AsyncClient(
                    base_url="http://model-service",
                    transport=httpx.ASGITransport(app=app),
                ) as client:
                    prediction = await client.models.classify_image(
                        "prostate-classifier-1",
                        np.zeros((128, 128, 3), dtype=np.uint8),
                    )
            self.assertEqual(prediction, 0.75)
        finally:
            serve.shutdown()
            ray.shutdown()


if __name__ == "__main__":
    unittest.main()
