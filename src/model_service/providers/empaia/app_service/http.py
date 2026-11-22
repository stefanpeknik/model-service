"""The App Service over HTTP through the EMPAIA SDK, with safe failure classes.

One method per SDK call family (`EmpaiaGateway`); converters and publishers
turn inputs and outputs into these calls. Writes run a durable guard first and
mark outcomes that may have applied as unknown.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import ClientError, ClientResponseError, ContentTypeError
from rationai.empaia import Client

from model_service.providers.empaia.app_service.gateway import (
    EmpaiaGateway,
    EmpaiaGatewayError,
    EmpaiaJobContext,
    EmpaiaJobMode,
)
from model_service.providers.empaia.app_service.responses import (
    ConfigurationResponse,
)
from model_service.providers.empaia.app_service.types import DirectAccess


if TYPE_CHECKING:
    from collections.abc import Awaitable

    from rationai.empaia.typing import SlideInfo
    from rationai.empaia.utils.scalar_output import PixelmapTileSet

    from model_service.contracts.json_values import JsonValue
    from model_service.providers.empaia.app_service.gateway import (
        EmpaiaGatewayFactory,
        MutationGuard,
    )
    from model_service.providers.empaia.settings import EmpaiaSettings


class HttpEmpaiaGateway(EmpaiaGateway):
    """Access one EMPAIA job through the SDK client."""

    def __init__(
        self,
        api_url: str,
        job_id: str,
        credential: str,
        *,
        client: Client | None = None,
    ) -> None:
        """Require complete connection settings and select the SDK client.

        close() closes even an injected client.

        Args:
            api_url: The App Service URL of the job.
            job_id: The EMPAIA job ID.
            credential: The job's token.
            client: The SDK client; one is built from the connection settings if omitted.

        Raises:
            EmpaiaGatewayError: If `api_url`, `job_id` or `credential` is empty; not
                retryable.
        """
        if not api_url or not job_id or not credential:
            raise EmpaiaGatewayError(
                "EMPAIA connection settings are incomplete",
                retryable=False,
            )
        self._job_id = job_id
        self._client = client or Client(api_url, job_id, credential)

    @property
    def job_id(self) -> str:
        """The EMPAIA job, which creates the outputs."""
        return self._job_id

    def set_before_mutation(self, callback: MutationGuard) -> None:
        """Install the async durable-claim guard used immediately before each external mutation.

        Args:
            callback: Awaited before each external write.
        """
        self._client.set_before_mutation(callback)

    async def get_configuration(self) -> dict[str, JsonValue]:
        """GET and validate job configuration, rejecting overlapping global/customer keys.

        Returns:
            The global and customer configuration, merged.

        Raises:
            EmpaiaGatewayError: Not retryable if the response is malformed or the
                sections share a key; otherwise as `_call` classifies a failed request.
        """
        value = await self._call(self._client.get_configuration())
        return ConfigurationResponse.parse(value).merged()

    async def get_mode(self) -> EmpaiaJobMode:
        """Fetch how the job was started.

        Returns:
            The job's mode.

        Raises:
            EmpaiaGatewayError: Not retryable if the mode is unknown; otherwise
                as `_call` classifies a failed request.
        """
        value = await self._call(self._client.get_mode())
        try:
            return EmpaiaJobMode(value)
        except ValueError as error:
            raise EmpaiaGatewayError.invalid("EMPAIA job mode is unknown") from error

    async def has_input(self, name: str) -> bool:
        """Check whether the job has the named input.

        AppService answers 404 for an input key the job does not have.

        Args:
            name: The EMPAIA input key.

        Returns:
            False on a 404, True if the input data was returned.

        Raises:
            EmpaiaGatewayError: For any failure other than a 404, as `_call` classifies a failed request.
        """
        try:
            await self._call(self._client.get_input_data(name))
        except EmpaiaGatewayError as error:
            cause = error.__cause__
            if isinstance(cause, ClientResponseError) and cause.status == 404:
                return False
            raise
        return True

    async def get_input_data(self, name: str) -> JsonValue:
        """GET the job's named input as the App Service gives it.

        Args:
            name: The EMPAIA input key.

        Returns:
            The decoded input data.

        Raises:
            EmpaiaGatewayError: As `_call` classifies a failed request.
        """
        return await self._call(self._client.get_input_data(name))

    async def direct_access(self, name: str) -> DirectAccess | None:
        """Query the RationAI direct-access route for the input's path on the slide volume.

        Args:
            name: The EMPAIA input key.

        Returns:
            The path and whether direct access is supported, or None if the
            input has no direct-access path.

        Raises:
            EmpaiaGatewayError: As `_call` classifies a failed request.
        """
        info = await self._call(self._client.get_direct_access_info(name))
        # CDS answers the path under its own data directory (physical_path)
        # and relative to it (storage_path); model-service mounts the slide
        # volume where CDS does, so the absolute path resolves here too.
        if info is None or not info.path:
            return None
        return DirectAccess(
            storage_path=Path(info.path),
            supported=info.direct_access_supported,
        )

    async def post_output(self, name: str, body: Any) -> None:
        """POST the guarded output body.

        Args:
            name: The EMPAIA output key.
            body: The SDK's request model of the output.

        Raises:
            EmpaiaGatewayError: As `_call` classifies a failed request as a mutation.
        """
        await self._call(self._client.post_output(name, body), mutation=True)

    async def finalize(self) -> None:
        """PUT the guarded successful job-finalization request.

        Raises:
            EmpaiaGatewayError: As `_call` classifies a failed request as a mutation.
        """
        await self._call(self._client.put_finalize(), mutation=True)

    async def report_progress(self, fraction: float) -> None:
        """PUT the job's progress.

        Args:
            fraction: Share of the work done, from 0 to 1.

        Raises:
            EmpaiaGatewayError: As `_call` classifies a failed request.
        """
        await self._call(self._client.put_progress(fraction))

    async def report_failure(self, message: str) -> None:
        """PUT a guarded failure request with the safe user message truncated to 255 characters.

        Args:
            message: The safe message to report.

        Raises:
            EmpaiaGatewayError: As `_call` classifies a failed request as a mutation.
        """
        await self._call(self._client.put_failure(message[:255]), mutation=True)

    async def close(self) -> None:
        """Close the gateway's HTTP client and release its network resources."""
        await self._client.close()

    async def slide_info(self, name: str) -> SlideInfo:
        """Fetch the slide with every level's downsample factor, as pixelmap tiles need.

        Args:
            name: The EMPAIA input key of the slide.

        Returns:
            The job's slide input.

        Raises:
            EmpaiaGatewayError: As `_call` classifies a failed request.
        """
        return await self._call(self._client.get_input(name))

    async def publish_pixelmap(
        self, name: str, slide_id: str, tile_sets: tuple[PixelmapTileSet, ...]
    ) -> None:
        """Create the pixelmap and upload its levels; failures leave an unknown outcome.

        Args:
            name: The EMPAIA output key, also the pixelmap's name.
            slide_id: The slide the pixelmap belongs to.
            tile_sets: The pyramid levels to upload.

        Raises:
            EmpaiaGatewayError: Not retryable with an unknown outcome if the SDK gives
                up on a tile or cannot write its storage directly; otherwise as `_call` classifies a failed request
                as a mutation.
        """
        try:
            await self._call(
                self._client.upload_pixelmap_pyramid(
                    key=name, name=name, wsi_id=slide_id, tile_sets=tile_sets
                ),
                mutation=True,
            )
        except (RuntimeError, OSError) as error:
            # The SDK gave up on a tile after its own retries, or could not write
            # its storage directly; the pixelmap exists.
            raise EmpaiaGatewayError(
                "EMPAIA pixelmap upload failed", retryable=False, outcome_unknown=True
            ) from error

    async def _call[T](self, operation: Awaitable[T], *, mutation: bool = False) -> T:
        """Await an SDK call, translating failures without exposing upstream details.

        Args:
            operation: The SDK call.
            mutation: Whether the call writes; a failure that may have applied it is
                marked `outcome_unknown`.

        Returns:
            The call's result.

        Raises:
            EmpaiaGatewayError: Retryable on a 429, a 5xx, a connection error or a
                timeout; not retryable on a 401 or 403 (rejected credential), another
                HTTP error, or a response that is not JSON or not the expected shape.
                A failed mutation's outcome is unknown, except after a 429 or another
                4xx.
        """
        try:
            return await operation
        except ContentTypeError as error:
            # The App Service answered 2xx with something that is not JSON.
            raise EmpaiaGatewayError(
                "EMPAIA AppService returned an invalid response",
                retryable=False,
                outcome_unknown=mutation,
            ) from error
        except ClientResponseError as error:
            if error.status == 429 or error.status >= 500:
                raise EmpaiaGatewayError(
                    "EMPAIA AppService is temporarily unavailable",
                    retryable=True,
                    # A 429 refused the request; a 5xx may have applied it.
                    outcome_unknown=mutation and error.status != 429,
                ) from error
            if error.status in {401, 403}:
                raise EmpaiaGatewayError.invalid(
                    "EMPAIA rejected the job credential"
                ) from error
            raise EmpaiaGatewayError.invalid(
                "EMPAIA rejected the application request"
            ) from error
        except (ClientError, TimeoutError) as error:
            raise EmpaiaGatewayError(
                "EMPAIA AppService is temporarily unavailable",
                retryable=True,
                outcome_unknown=mutation,
            ) from error
        except (ValueError, TypeError) as error:
            # Not JSON, or not the shape the SDK's models require.
            raise EmpaiaGatewayError(
                "EMPAIA AppService returned an invalid response",
                retryable=False,
                outcome_unknown=mutation,
            ) from error


def http_gateway_factory(settings: EmpaiaSettings) -> EmpaiaGatewayFactory:
    """Build the gateway factory of a deployment, configured once from its settings.

    Args:
        settings: Supplies the App Service timeout.

    Returns:
        A factory creating one `HttpEmpaiaGateway` per job.
    """

    def create(context: EmpaiaJobContext) -> HttpEmpaiaGateway:
        """Create the SDK-backed gateway for one EMPAIA job.

        Args:
            context: The job's App Service URL, id and credential.

        Returns:
            The gateway.
        """
        return HttpEmpaiaGateway(
            context.api_url,
            context.job_id,
            context.credential,
            client=Client(
                context.api_url,
                context.job_id,
                context.credential,
                timeout=settings.app_service_timeout_seconds,
            ),
        )

    return create
