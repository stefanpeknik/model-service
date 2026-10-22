"""Check AppService HTTP shapes, input materialization, and uncertain publication outcomes."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from aiohttp import web
from aiohttp.test_utils import TestServer
from rationai.empaia.app_service import AppServiceClient

from model_service.builders.contracts import InlineOutput, SharedArtifact
from model_service.providers.empaia.gateway import EmpaiaGatewayError
from model_service.providers.empaia.http_gateway import (
    HttpEmpaiaGateway,
    Rectangle,
    SlideAccess,
    SlideMetadata,
)


class RecordingArtifactPublisher:
    """Record shared artifacts passed from the HTTP gateway without performing image conversion."""

    def __init__(self) -> None:
        """Start an empty list of named artifact publication requests."""
        self.calls: list[tuple[str, SharedArtifact]] = []

    async def publish(
        self,
        gateway: HttpEmpaiaGateway,
        name: str,
        artifact: SharedArtifact,
    ) -> None:
        """Record the output name and artifact without writing to an external service."""
        del gateway
        self.calls.append((name, artifact))


class RecordingRectangleWriter:
    """Record rectangle-mask requests and create placeholder destination files."""

    def __init__(self) -> None:
        """Start an empty list of mask destinations and geometry requests."""
        self.calls: list[tuple[Path, SlideMetadata, Rectangle]] = []

    def write(
        self,
        destination: Path,
        slide: SlideMetadata,
        rectangle: Rectangle,
    ) -> None:
        """Record requested slide/rectangle geometry and create a placeholder mask file."""
        self.calls.append((destination, slide, rectangle))
        destination.touch()


class FakeVipsImage:
    """Stand in for libvips images so tests can observe encoding or publication without native image I/O."""

    format = "uchar"
    width = 600
    height = 300
    bands = 1

    def crop(self, _x: int, _y: int, _width: int, _height: int) -> FakeVipsImage:
        """Return the same fake image instead of executing native cropping."""
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
        """Require black padding and return the fake image for edge-tile publication."""
        assert extend == "black"
        return self

    def cast(self, image_format: str) -> FakeVipsImage:
        """Accept the requested conversion without native pixel processing."""
        assert image_format == "float"
        return self

    def __truediv__(self, value: float) -> FakeVipsImage:
        """Require normalization by 255 and return the same fake image."""
        assert value == 255.0
        return self

    def write_to_memory(self) -> memoryview:
        """Return deterministic placeholder bytes for the fake float32 tile upload."""
        return memoryview(b"float32 tile")


class FakeVipsImageType:
    """Provide the fake libvips file-open operation used by publication tests."""

    @staticmethod
    def new_from_file(_path: str, *, access: str) -> FakeVipsImage:
        """Require sequential file access and return the fake image fixture."""
        assert access == "sequential"
        return FakeVipsImage()


class HttpEmpaiaGatewayTests(unittest.IsolatedAsyncioTestCase):
    """Exercise AppService HTTP shapes, input materialization, and uncertain publication outcomes."""

    async def appservice(self, transport: httpx.MockTransport) -> AppServiceClient:
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
        client = AppServiceClient(str(server.make_url("/")), "job-1", "credential")
        self.addAsyncCleanup(client.close)
        return client

    async def test_invalid_rectangle_geometry_is_a_safe_integration_error(self) -> None:
        """Verify invalid rectangle geometry is a safe integration error."""
        for coordinate in [float("nan"), float("inf"), True, "10"]:
            with self.subTest(coordinate=coordinate), TemporaryDirectory() as directory:
                writer = RecordingRectangleWriter()

                def handle(
                    request: httpx.Request, coordinate=coordinate
                ) -> httpx.Response:
                    """Return scenario-specific mock HTTP responses and record or assert requests for this gateway test."""
                    if request.url.path.endswith("/direct-access"):
                        return httpx.Response(404)
                    if request.url.path.endswith("/inputs/slide"):
                        return httpx.Response(
                            200,
                            json={
                                "id": "slide-1",
                                "extent": {"x": 1200, "y": 600, "z": 1},
                                "pixel_size_nm": {"x": 250, "y": 250},
                                "levels": [{"extent": {"x": 1200, "y": 600}}],
                            },
                        )
                    return httpx.Response(
                        200,
                        content=json.dumps(
                            {
                                "upper_left": [coordinate, 20],
                                "width": 100,
                                "height": 50,
                            }
                        ),
                    )

                gateway = HttpEmpaiaGateway(
                    "https://empaia.example",
                    "job-1",
                    "credential",
                    client=await self.appservice(httpx.MockTransport(handle)),
                    rectangle_writer=writer,
                )
                try:
                    with self.assertRaises(EmpaiaGatewayError) as raised:
                        await gateway.materialize_input("roi", directory)
                    self.assertFalse(raised.exception.retryable)
                    self.assertEqual(writer.calls, [])
                finally:
                    await gateway.close()

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

    async def test_configuration_is_flattened_and_direct_input_is_linked(self) -> None:
        """Verify configuration is flattened and direct input is linked."""
        with TemporaryDirectory() as directory:
            source = Path(directory) / "slide.tiff"
            source.write_bytes(b"slide")

            def handle(request: httpx.Request) -> httpx.Response:
                """Return scenario-specific mock HTTP responses and record or assert requests for this gateway test."""
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

            client = await self.appservice(httpx.MockTransport(handle))
            gateway = HttpEmpaiaGateway(
                "https://empaia.example",
                "job-1",
                "credential",
                client=client,
            )
            workspace = Path(directory) / "workspace"

            configuration = await gateway.get_configuration()
            destination = Path(await gateway.materialize_input("slide", str(workspace)))

            self.assertEqual(configuration, {"stride": 0.5, "tiles": 8})
            self.assertEqual(destination, workspace / source.name)
            self.assertTrue(destination.is_symlink())
            self.assertEqual(destination.read_bytes(), b"slide")
            await gateway.close()

    async def test_inline_outputs_and_lifecycle_use_appservice_contract(self) -> None:
        """Verify inline outputs and lifecycle use AppService contract."""
        requests: list[tuple[str, str, object | None]] = []

        def handle(request: httpx.Request) -> httpx.Response:
            """Return scenario-specific mock HTTP responses and record or assert requests for this gateway test."""
            body = json.loads(request.content) if request.content else None
            requests.append((request.method, request.url.path, body))
            return httpx.Response(200, json={})

        gateway = HttpEmpaiaGateway(
            "https://empaia.example",
            "job-1",
            "credential",
            client=await self.appservice(httpx.MockTransport(handle)),
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
        """Verify rectangle input is materialized when direct access is absent."""
        writer = RecordingRectangleWriter()

        def handle(request: httpx.Request) -> httpx.Response:
            """Return scenario-specific mock HTTP responses and record or assert requests for this gateway test."""
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
            gateway = HttpEmpaiaGateway(
                "https://empaia.example",
                "job-1",
                "credential",
                client=await self.appservice(httpx.MockTransport(handle)),
                rectangle_writer=writer,
            )

            destination = Path(
                await gateway.materialize_input("region_of_interest", directory)
            )

            self.assertEqual(destination, Path(directory) / "mask.tiff")
            self.assertTrue(destination.is_file())
            self.assertEqual(len(writer.calls), 1)
            _, slide, rectangle = writer.calls[0]
            self.assertEqual(slide.identifier, "slide-1")
            self.assertEqual(rectangle, Rectangle(10, 20, 100, 50))
            await gateway.close()

    async def test_http_slide_access_downloads_slides_and_still_writes_masks(
        self,
    ) -> None:
        """Verify HTTP slide access skips direct access, downloads slides, and keeps masks."""
        requested: list[str] = []
        downloads: list[tuple[SlideMetadata, float, Path]] = []
        slide = {
            "id": "slide-1",
            "extent": {"x": 1200, "y": 600, "z": 1},
            "pixel_size_nm": {"x": 250.0, "y": 250.0},
            "levels": [{"extent": {"x": 1200, "y": 600, "z": 1}}],
        }

        class RecordingDownloader:
            async def download(
                self,
                gateway: HttpEmpaiaGateway,
                metadata: SlideMetadata,
                mpp: float,
                destination: Path,
            ) -> None:
                del gateway
                downloads.append((metadata, mpp, destination))
                destination.touch()

        def handle(request: httpx.Request) -> httpx.Response:
            requested.append(request.url.path)
            if request.url.path.endswith("/inputs/slide"):
                return httpx.Response(200, json=slide)
            if request.url.path.endswith("/inputs/region_of_interest"):
                return httpx.Response(
                    200, json={"upper_left": [10, 20], "width": 100, "height": 50}
                )
            raise AssertionError(request.url.path)

        writer = RecordingRectangleWriter()
        with TemporaryDirectory() as directory:
            gateway = HttpEmpaiaGateway(
                "https://empaia.example",
                "job-1",
                "credential",
                client=await self.appservice(httpx.MockTransport(handle)),
                rectangle_writer=writer,
                slide_access=SlideAccess.HTTP,
                slide_downloader=RecordingDownloader(),
            )
            slide_path = await gateway.materialize_input("slide", directory, mpp=0.5)
            mask_path = await gateway.materialize_input(
                "region_of_interest", directory, mpp=0.5
            )
            with self.assertRaises(EmpaiaGatewayError) as raised:
                await gateway.materialize_input("slide", directory)
            await gateway.close()

        self.assertEqual(slide_path, f"{directory}/slide.tiff")
        self.assertEqual(mask_path, f"{directory}/mask.tiff")
        self.assertEqual(len(downloads), 1)
        self.assertEqual(downloads[0][0].identifier, "slide-1")
        self.assertEqual(downloads[0][1], 0.5)
        self.assertEqual(len(writer.calls), 1)
        self.assertFalse(raised.exception.retryable)
        self.assertFalse(any(path.endswith("/direct-access") for path in requested))

    async def test_region_reads_retry_transient_failures(self) -> None:
        """Verify a region read retries a transient AppService failure."""
        answers = [httpx.Response(503), httpx.Response(200, content=b"png")]

        def handle(request: httpx.Request) -> httpx.Response:
            self.assertEqual(
                request.url.path,
                "/v3/job-1/regions/slide-1/level/1/start/512/0/size/256/128",
            )
            return answers.pop(0)

        gateway = HttpEmpaiaGateway(
            "https://empaia.example",
            "job-1",
            "credential",
            client=await self.appservice(httpx.MockTransport(handle)),
        )
        with patch(
            "model_service.providers.empaia.http_gateway._READ_RETRY_SECONDS", 0
        ):
            data = await gateway.read_region("slide-1", 1, 512, 0, 256, 128)
        await gateway.close()

        self.assertEqual(data, b"png")
        self.assertEqual(answers, [])

    async def test_shared_artifact_uses_packaged_publisher(self) -> None:
        """Verify shared artifact uses packaged publisher."""
        publisher = RecordingArtifactPublisher()
        gateway = HttpEmpaiaGateway(
            "https://empaia.example",
            "job-1",
            "credential",
            client=await self.appservice(
                httpx.MockTransport(lambda _request: httpx.Response(200, json={}))
            ),
            artifact_publisher=publisher,
        )
        artifact = SharedArtifact(path="/shared/heatmap.tiff", media_type="image/tiff")

        await gateway.publish_output("heatmap", artifact)

        self.assertEqual(publisher.calls, [("heatmap", artifact)])
        await gateway.close()

    async def test_default_tiff_publisher_registers_and_uploads_pixelmap(self) -> None:
        """Verify default TIFF publisher registers and uploads pixelmap."""
        uploaded_tiles: list[str] = []

        def handle(request: httpx.Request) -> httpx.Response:
            """Return scenario-specific mock HTTP responses and record or assert requests for this gateway test."""
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
                client=await self.appservice(httpx.MockTransport(handle)),
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
        """Verify mutating transport failure is safe and outcome unknown."""

        def fail(_request: httpx.Request) -> httpx.Response:
            """Inject the configured remote failure for safe classification or terminal reporting checks."""
            raise httpx.ConnectError("private transport detail")

        gateway = HttpEmpaiaGateway(
            "https://empaia.example",
            "job-1",
            "credential",
            client=await self.appservice(httpx.MockTransport(fail)),
        )

        with self.assertRaises(EmpaiaGatewayError) as raised:
            await gateway.finalize()

        self.assertTrue(raised.exception.retryable)
        self.assertTrue(raised.exception.outcome_unknown)
        self.assertNotIn("private transport detail", raised.exception.safe_message)
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
