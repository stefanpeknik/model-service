from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from integrations.empaia.gateway import EmpaiaGatewayError
from integrations.empaia.http_gateway import (
    HttpEmpaiaGateway,
    Rectangle,
    SlideMetadata,
)
from workflow_execution.contracts import InlineOutput, SharedArtifact


class RecordingArtifactPublisher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, SharedArtifact]] = []

    async def publish(
        self,
        gateway: HttpEmpaiaGateway,
        name: str,
        artifact: SharedArtifact,
    ) -> None:
        del gateway
        self.calls.append((name, artifact))


class RecordingRectangleWriter:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, SlideMetadata, Rectangle]] = []

    def write(
        self,
        destination: Path,
        slide: SlideMetadata,
        rectangle: Rectangle,
    ) -> None:
        self.calls.append((destination, slide, rectangle))
        destination.touch()


class FakeVipsImage:
    format = "uchar"
    width = 600
    height = 300
    bands = 1

    def crop(self, _x: int, _y: int, _width: int, _height: int) -> FakeVipsImage:
        return self

    def embed(
        self,
        _x: int,
        _y: int,
        _width: int,
        _height: int,
        *,
        extend: str,
    ) -> FakeVipsImage:
        assert extend == "black"
        return self

    def cast(self, image_format: str) -> FakeVipsImage:
        assert image_format == "float"
        return self

    def __truediv__(self, value: float) -> FakeVipsImage:
        assert value == 255.0
        return self

    def write_to_memory(self) -> memoryview:
        return memoryview(b"float32 tile")


class FakeVipsImageType:
    @staticmethod
    def new_from_file(_path: str, *, access: str) -> FakeVipsImage:
        assert access == "sequential"
        return FakeVipsImage()


class HttpEmpaiaGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_configuration_is_flattened_and_direct_input_is_linked(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "slide.tiff"
            source.write_bytes(b"slide")

            def handle(request: httpx.Request) -> httpx.Response:
                if request.url.path.endswith("/configuration"):
                    return httpx.Response(
                        200,
                        json={"global": {"stride": 0.5}, "customer": {"tiles": 8}},
                    )
                if request.url.path.endswith("/direct-access"):
                    return httpx.Response(
                        200,
                        json={
                            "slide_id": "slide-1",
                            "storage_path": str(source),
                            "direct_access_supported": True,
                        },
                    )
                raise AssertionError(request.url.path)

            client = httpx.AsyncClient(
                transport=httpx.MockTransport(handle),
                base_url="https://empaia.example",
            )
            gateway = HttpEmpaiaGateway(
                "https://empaia.example",
                "job-1",
                "credential",
                client=client,
            )
            destination = Path(directory) / "workspace" / "slide.tiff"

            configuration = await gateway.get_configuration()
            await gateway.materialize_input("slide", str(destination))

            self.assertEqual(configuration, {"stride": 0.5, "tiles": 8})
            self.assertTrue(destination.is_symlink())
            self.assertEqual(destination.read_bytes(), b"slide")
            await gateway.close()

    async def test_inline_outputs_and_lifecycle_use_appservice_contract(self) -> None:
        requests: list[tuple[str, str, object | None]] = []

        def handle(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content) if request.content else None
            requests.append((request.method, request.url.path, body))
            return httpx.Response(200, json={})

        gateway = HttpEmpaiaGateway(
            "https://empaia.example",
            "job-1",
            "credential",
            client=httpx.AsyncClient(
                transport=httpx.MockTransport(handle),
                base_url="https://empaia.example",
            ),
        )

        await gateway.publish_output("score", InlineOutput(value=0.75))
        await gateway.finalize()
        await gateway.report_failure("failed safely")
        await gateway.close()

        self.assertEqual(
            requests,
            [
                (
                    "POST",
                    "/v3/job-1/outputs/score",
                    {
                        "name": "score",
                        "type": "float",
                        "value": 0.75,
                        "creator_id": "job-1",
                        "creator_type": "job",
                    },
                ),
                ("PUT", "/v3/job-1/finalize", None),
                (
                    "PUT",
                    "/v3/job-1/failure",
                    {"user_message": "failed safely"},
                ),
            ],
        )

    async def test_rectangle_input_is_materialized_when_direct_access_is_absent(
        self,
    ) -> None:
        writer = RecordingRectangleWriter()

        def handle(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/direct-access"):
                return httpx.Response(404)
            if request.url.path.endswith("/inputs/slide"):
                return httpx.Response(
                    200,
                    json={
                        "id": "slide-1",
                        "extent": {"x": 1200, "y": 600, "z": 1},
                        "pixel_size_nm": {"x": 250.0, "y": 250.0},
                        "levels": [{"extent": {"x": 1200, "y": 600, "z": 1}}],
                    },
                )
            if request.url.path.endswith("/inputs/region_of_interest"):
                return httpx.Response(
                    200,
                    json={"upper_left": [10.2, 20.8], "width": 100, "height": 50},
                )
            raise AssertionError(request.url.path)

        with TemporaryDirectory() as directory:
            destination = Path(directory) / "roi.tiff"
            gateway = HttpEmpaiaGateway(
                "https://empaia.example",
                "job-1",
                "credential",
                client=httpx.AsyncClient(
                    transport=httpx.MockTransport(handle),
                    base_url="https://empaia.example",
                ),
                rectangle_writer=writer,
            )

            await gateway.materialize_input(
                "region_of_interest",
                str(destination),
            )

            self.assertTrue(destination.is_file())
            self.assertEqual(len(writer.calls), 1)
            _, slide, rectangle = writer.calls[0]
            self.assertEqual(slide.identifier, "slide-1")
            self.assertEqual(rectangle, Rectangle(10, 20, 100, 50))
            await gateway.close()

    async def test_shared_artifact_uses_packaged_publisher(self) -> None:
        publisher = RecordingArtifactPublisher()
        gateway = HttpEmpaiaGateway(
            "https://empaia.example",
            "job-1",
            "credential",
            client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda _request: httpx.Response(200, json={})
                ),
                base_url="https://empaia.example",
            ),
            artifact_publisher=publisher,
        )
        artifact = SharedArtifact(path="/shared/heatmap.tiff", media_type="image/tiff")

        await gateway.publish_output("heatmap", artifact)

        self.assertEqual(publisher.calls, [("heatmap", artifact)])
        await gateway.close()

    async def test_default_tiff_publisher_registers_and_uploads_pixelmap(self) -> None:
        uploaded_tiles: list[str] = []

        def handle(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/inputs/slide"):
                return httpx.Response(
                    200,
                    json={
                        "id": "slide-1",
                        "extent": {"x": 1200, "y": 600, "z": 1},
                        "pixel_size_nm": {"x": 250.0, "y": 250.0},
                        "levels": [
                            {"extent": {"x": 1200, "y": 600, "z": 1}},
                            {"extent": {"x": 600, "y": 300, "z": 1}},
                        ],
                    },
                )
            if request.method == "POST":
                body = json.loads(request.content)
                self.assertEqual(body["levels"][0]["slide_level"], 1)
                self.assertEqual(body["levels"][0]["position_max_x"], 1)
                return httpx.Response(200, json={"id": "pixelmap-1"})
            if request.method == "PUT" and "/pixelmaps/" in request.url.path:
                uploaded_tiles.append(request.url.path)
                self.assertEqual(request.content, b"float32 tile")
                return httpx.Response(204)
            raise AssertionError((request.method, request.url.path))

        with TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "heatmap.tiff"
            artifact_path.touch()
            gateway = HttpEmpaiaGateway(
                "https://empaia.example",
                "job-1",
                "credential",
                client=httpx.AsyncClient(
                    transport=httpx.MockTransport(handle),
                    base_url="https://empaia.example",
                ),
            )

            with patch.dict(
                "sys.modules",
                {"pyvips": SimpleNamespace(Image=FakeVipsImageType)},
            ):
                await gateway.publish_output(
                    "heatmap",
                    SharedArtifact(
                        path=str(artifact_path),
                        media_type="image/tiff",
                    ),
                )

            self.assertEqual(len(uploaded_tiles), 2)
            await gateway.close()

    async def test_mutating_transport_failure_is_safe_and_outcome_unknown(self) -> None:
        def fail(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("private transport detail")

        gateway = HttpEmpaiaGateway(
            "https://empaia.example",
            "job-1",
            "credential",
            client=httpx.AsyncClient(
                transport=httpx.MockTransport(fail),
                base_url="https://empaia.example",
            ),
        )

        with self.assertRaises(EmpaiaGatewayError) as raised:
            await gateway.finalize()

        self.assertTrue(raised.exception.retryable)
        self.assertTrue(raised.exception.outcome_unknown)
        self.assertNotIn("private transport detail", raised.exception.safe_message)
        await gateway.close()

    async def test_configuration_rejects_ambiguous_sections(self) -> None:
        gateway = HttpEmpaiaGateway(
            "https://empaia.example",
            "job-1",
            "credential",
            client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda _request: httpx.Response(
                        200,
                        json={"global": {"tiles": 8}, "customer": {"tiles": 16}},
                    )
                ),
                base_url="https://empaia.example",
            ),
        )

        with self.assertRaises(EmpaiaGatewayError) as raised:
            await gateway.get_configuration()

        self.assertFalse(raised.exception.retryable)
        await gateway.close()


if __name__ == "__main__":
    unittest.main()
