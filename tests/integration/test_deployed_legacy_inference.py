"""Opt-in checks against real models through the public scheduler and SDK.

Set RUN_DEPLOYED_LEGACY_INFERENCE=1 and VALIDATION_ORIGIN. Select one model
with LEGACY_MODEL_ROUTES, or leave it unset to check all four models sequentially.
These checks cover numerical consistency and transport. Only the classifier
zero-tile score has a stored reference; no independent artifact accuracy is
established for segmentation or embeddings.
"""

from __future__ import annotations

import asyncio
import os
import unittest


_MODEL_ROUTES = frozenset(
    {"prostate-classifier-1", "episeg-1", "prov-gigapath", "virchow2"}
)


@unittest.skipUnless(
    os.environ.get("RUN_DEPLOYED_LEGACY_INFERENCE") == "1",
    "set RUN_DEPLOYED_LEGACY_INFERENCE=1 to call deployed legacy models",
)
class DeployedLegacyInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.origin = os.environ.get("VALIDATION_ORIGIN")
        if not self.origin:
            self.fail("VALIDATION_ORIGIN must point at the public scheduler")
        selected = os.environ.get(
            "LEGACY_MODEL_ROUTES", ",".join(sorted(_MODEL_ROUTES))
        )
        self.routes = {
            route.strip().strip("/") for route in selected.split(",") if route.strip()
        }
        if not self.routes or self.routes - _MODEL_ROUTES:
            self.fail(
                "LEGACY_MODEL_ROUTES must select comma-separated model names from "
                + ", ".join(sorted(_MODEL_ROUTES))
            )

    def _select(self, route: str) -> None:
        if route not in self.routes:
            self.skipTest(f"{route} is not selected in LEGACY_MODEL_ROUTES")

    @staticmethod
    def _tile(size: int):
        import numpy as np

        y, x = np.indices((size, size), dtype=np.uint16)
        return np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)

    def test_classifier_reference_and_sdk_image_consistency(self) -> None:
        self._select("prostate-classifier-1")
        import lz4.frame
        import numpy as np
        from PIL import Image
        from rationai import AsyncClient, Client

        zero = np.zeros((512, 512, 3), dtype=np.uint8)
        tile = self._tile(512)
        with Client(base_url=self.origin, timeout=900) as client:
            score = client.models.classify_image("prostate-classifier-1", zero)
            self.assertAlmostEqual(score, 0.256591796875, delta=0.001)
            direct = client.post(
                "prostate-classifier-1/",
                content=lz4.frame.compress(zero.tobytes()),
                headers={"content-type": "application/octet-stream"},
            )
            direct.raise_for_status()
            self.assertAlmostEqual(direct.json(), score, delta=0.001)
            array_score = client.models.classify_image("prostate-classifier-1", tile)
            pil_score = client.models.classify_image(
                "prostate-classifier-1", Image.fromarray(tile)
            )

        async def classify():
            async with AsyncClient(base_url=self.origin, timeout=900) as client:
                return await client.models.classify_image("prostate-classifier-1", tile)

        async_score = asyncio.run(classify())
        for result in (array_score, pil_score, async_score):
            self.assertTrue(np.isfinite(result))
            self.assertGreaterEqual(result, 0)
            self.assertLessEqual(result, 1)
            self.assertAlmostEqual(result, array_score, delta=0.001)

    def test_segmentation_tensor_and_sdk_image_consistency(self) -> None:
        self._select("episeg-1")
        import numpy as np
        from PIL import Image
        from rationai import AsyncClient, Client

        tile = self._tile(1024)
        with Client(base_url=self.origin, timeout=900) as client:
            array_result = client.models.segment_image("episeg-1", tile)
            pil_result = client.models.segment_image("episeg-1", Image.fromarray(tile))
            repeated_result = client.models.segment_image("episeg-1", tile)

        async def segment():
            async with AsyncClient(base_url=self.origin, timeout=900) as client:
                return await client.models.segment_image("episeg-1", tile)

        async_result = asyncio.run(segment())
        for result in (
            array_result,
            pil_result,
            repeated_result,
            async_result,
        ):
            self.assertEqual(result.shape, (1, 1024, 1024))
            self.assertEqual(result.dtype, np.dtype(np.float16))
            self.assertTrue(np.isfinite(result).all())
            np.testing.assert_allclose(result, array_result, rtol=0.001, atol=0.001)

    def _embedding_results(self, route: str, **headers: str):
        import lz4.frame
        import numpy as np
        from PIL import Image
        from rationai import AsyncClient, Client

        tile = self._tile(224)
        with Client(base_url=self.origin, timeout=900) as client:
            full_precision = client.models.embed_image(route, tile, **headers)
            half_precision = client.models.embed_image(
                route, tile, output_dtype=np.float16, **headers
            )
            pil_result = client.models.embed_image(
                route, Image.fromarray(tile), **headers
            )
            wire = client.post(
                f"{route}/",
                content=lz4.frame.compress(tile.tobytes()),
                headers={
                    "content-type": "application/octet-stream",
                    "x-output-dtype": "float32",
                    **{
                        f"x-{name.replace('_', '-')}": value
                        for name, value in headers.items()
                    },
                },
            )
            wire.raise_for_status()
            self.assertEqual(
                wire.headers["content-type"].split(";")[0],
                "application/octet-stream",
            )
            self.assertEqual(wire.headers["x-output-shape"], str(full_precision.shape))
            self.assertEqual(
                len(lz4.frame.decompress(wire.content)), full_precision.nbytes
            )

        async def embed():
            async with AsyncClient(base_url=self.origin, timeout=900) as client:
                return await client.models.embed_image(route, tile, **headers)

        async_result = asyncio.run(embed())
        self.assertGreater(full_precision.size, 0)
        self.assertGreater(full_precision.ndim, 0)
        for result, dtype in (
            (full_precision, np.float32),
            (half_precision, np.float16),
            (pil_result, np.float32),
            (async_result, np.float32),
        ):
            self.assertEqual(result.shape, full_precision.shape)
            self.assertEqual(result.dtype, np.dtype(dtype))
            self.assertTrue(np.isfinite(result).all())
            np.testing.assert_allclose(
                result.astype(np.float32),
                full_precision,
                rtol=0.002,
                atol=0.002,
            )
        return full_precision

    def test_gigapath_embedding_precision_and_sdk_image_consistency(self) -> None:
        self._select("prov-gigapath")
        self._embedding_results("prov-gigapath")

    def test_virchow_embedding_pooling_precision_and_sdk_image_consistency(
        self,
    ) -> None:
        self._select("virchow2")
        import numpy as np

        pooled = self._embedding_results("virchow2")
        unpooled = self._embedding_results("virchow2", pool_tokens="false")
        self.assertEqual(pooled.ndim, 2)
        self.assertEqual(unpooled.ndim, 2)
        self.assertEqual(pooled.shape, (2, unpooled.shape[1]))
        self.assertGreater(unpooled.shape[0], 2)
        np.testing.assert_allclose(pooled[0], unpooled[0], rtol=0.002, atol=0.002)
        np.testing.assert_allclose(
            pooled[1], unpooled[1:].mean(axis=0), rtol=0.002, atol=0.002
        )


if __name__ == "__main__":
    unittest.main()
