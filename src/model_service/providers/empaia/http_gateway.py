"""Implement AppService HTTP access and classify safe integration failures.

Output publication invokes a durable guard before writes and tracks uncertain
mutation outcomes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import ClientError, ClientResponseError, ContentTypeError
from rationai.empaia import Client
from rationai.empaia.typing import primitives

from model_service.providers.empaia.gateway import (
    EmpaiaGateway,
    EmpaiaGatewayError,
    EmpaiaJobContext,
    EmpaiaJobMode,
    PrimitiveType,
)
from model_service.providers.empaia.responses import ConfigurationResponse


if TYPE_CHECKING:
    from collections.abc import Awaitable

    from model_service.builders.contracts import JsonValue
    from model_service.providers.empaia.gateway import (
        EmpaiaGatewayFactory,
        MutationGuard,
    )
    from model_service.providers.empaia.settings import EmpaiaSettings


class HttpEmpaiaGateway(EmpaiaGateway):
    """Access one EMPAIA job through authenticated HTTP."""

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

    async def post_primitive(
        self, name: str, output_type: str, value: JsonValue
    ) -> None:
        """POST the guarded primitive output.

        Args:
            name: The EMPAIA output key.
            output_type: The EMPAIA primitive type, e.g. `integer`.
            value: The output's value.

        Raises:
            EmpaiaGatewayError: Not retryable if the type has no primitive or the value
                does not fit it; otherwise as `_call` classifies a failed request as a mutation.
        """
        await self._call(
            self._client.post_output(
                name, _primitive_output(name, self._job_id, output_type, value)
            ),
            mutation=True,
        )

    async def finalize(self) -> None:
        """PUT the guarded successful job-finalization request.

        Raises:
            EmpaiaGatewayError: As `_call` classifies a failed request as a mutation.
        """
        await self._call(self._client.put_finalize(), mutation=True)

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


type _PostPrimitive = (
    primitives.PostBoolPrimitive
    | primitives.PostIntegerPrimitive
    | primitives.PostFloatPrimitive
    | primitives.PostStringPrimitive
)


# The SDK's request model for each primitive type.
_PRIMITIVES: dict[PrimitiveType, type[_PostPrimitive]] = {
    PrimitiveType.BOOL: primitives.PostBoolPrimitive,
    PrimitiveType.INTEGER: primitives.PostIntegerPrimitive,
    PrimitiveType.FLOAT: primitives.PostFloatPrimitive,
    PrimitiveType.STRING: primitives.PostStringPrimitive,
}


def _primitive_output(
    name: str, job_id: str, output_type: str, value: JsonValue
) -> _PostPrimitive:
    """Encode a primitive of the given type with EMPAIA output and creator metadata.

    Args:
        name: The EMPAIA output key.
        job_id: The job, recorded as the creator.
        output_type: The EMPAIA primitive type, e.g. `integer`.
        value: The output's value.

    Returns:
        The primitive to post.

    Raises:
        EmpaiaGatewayError: Not retryable if EMPAIA has no such primitive type or
            the value does not fit it.
    """
    try:
        kind = _PRIMITIVES[PrimitiveType(output_type)]
    except ValueError as error:
        raise EmpaiaGatewayError.invalid(
            f"EMPAIA has no {output_type!r} primitive"
        ) from error
    try:
        return kind.model_validate(
            {
                "name": name,
                "type": output_type,
                "value": value,
                "creator_id": job_id,
                "creator_type": "job",
            }
        )
    except ValueError as error:
        raise EmpaiaGatewayError.invalid(
            "EMPAIA inline output is not a valid primitive"
        ) from error
