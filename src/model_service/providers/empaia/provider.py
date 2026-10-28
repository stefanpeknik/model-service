"""Translate EMPAIA jobs into durable executions and publish their stored outcomes.

Admission fetches configuration and caps work/reporting deadlines at credential expiry.
Finish retries are separate from computation and stop when publication's external
outcome is uncertain.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import shutil
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Never

import jwt
from pydantic import Field, ValidationError

from model_service.builders.application import validate_invocation
from model_service.builders.contracts import (
    InputAccessContext,
    InputAccessKind,
    InvalidInvocation,
    Invocation,
    ProviderInput,
    is_json_value,
)
from model_service.builders.registry import get_builder
from model_service.providers.empaia.gateway import (
    EmpaiaGateway,
    EmpaiaGatewayError,
    EmpaiaGatewayFactory,
    EmpaiaInputState,
    EmpaiaJobContext,
    EmpaiaJobMode,
)
from model_service.providers.empaia.input_access import job_workspace
from model_service.scheduler.contracts import (
    InvalidScheduleRequestError,
    ProviderName,
    ScheduleRequest,
    SchedulingUnavailableError,
)
from model_service.scheduler.domain import (
    PreparedInvocation,
    Priority,
    SubmissionIdentity,
)
from model_service.scheduler.providers import (
    ProviderCancellation,
    ProviderFailure,
    ProviderFinishFailureError,
    ProviderFinishOutcome,
    ProviderSuccess,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from model_service.builders.contracts import JsonValue
    from model_service.scheduler.providers import BeforeMutation


class EmpaiaErrorCode(StrEnum):
    """Name the safe failure for unsuccessful EMPAIA publication or terminal reporting."""

    FINISH_FAILED = "empaia_finish_failed"


class EmpaiaSchedulePayload(EmpaiaJobContext):
    """Require an installation ID alongside the job origin, ID, and private credential."""

    installation_id: Annotated[str, Field(min_length=1)]


logger = logging.getLogger(__name__)


class EmpaiaProviderAdapter:
    """Implement provider-scoped admission and terminal publication for EMPAIA jobs."""

    name = ProviderName.EMPAIA

    def __init__(
        self,
        gateway_factory: EmpaiaGatewayFactory,
        *,
        allowed_api_urls: tuple[str, ...],
        workspace_root: Path | None = None,
    ) -> None:
        """Only `allowed_api_urls` origins or path prefixes receive job credentials; none, no job."""
        self._gateway_factory = gateway_factory
        self._allowed_api_urls = tuple(url.rstrip("/") for url in allowed_api_urls)
        self._workspace_root = workspace_root

    def identify(self, request: ScheduleRequest) -> SubmissionIdentity:
        """Key admission by installation and job, fingerprinting stable request fields while excluding renewable credentials."""
        context = _schedule_payload(request.body)
        canonical = json.dumps(
            {
                "builder_id": request.builder_id,
                "model_id": request.model_id,
                "installation_id": context.installation_id,
                "job_id": context.job_id,
                "api_url": context.api_url,
                "requested_timeout_seconds": request.requested_timeout_seconds,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return SubmissionIdentity(
            provider=self.name,
            key=f"{context.installation_id}:{context.job_id}",
            fingerprint=hashlib.sha256(canonical).hexdigest(),
        )

    async def prepare(
        self,
        request: ScheduleRequest,
        deadline_at: datetime,
    ) -> PreparedInvocation:
        """Fetch and resolve configuration, record the job's mode, create provider input references, and separate public job state from protected credentials. Cap deadlines at credential expiry."""
        context = _schedule_payload(request.body)
        if not _allowed(context.api_url, self._allowed_api_urls):
            raise InvalidScheduleRequestError(
                "EMPAIA api_url is not an allowed AppService"
            )
        credential_expires_at = _credential_expiry(context.credential)
        async with _preparation_gateway(
            self._gateway_factory,
            context,
        ) as gateway:
            try:
                configuration = await gateway.get_configuration()
            except EmpaiaGatewayError as error:
                if error.retryable:
                    raise SchedulingUnavailableError(error.safe_message) from error
                raise InvalidScheduleRequestError(error.safe_message) from error
            if not is_json_value(configuration):
                raise InvalidScheduleRequestError(
                    "EMPAIA configuration must contain only JSON-compatible values"
                )
            try:
                mode = await gateway.get_mode()
            except EmpaiaGatewayError as error:
                _raise_preparation_error(error)
            spec = get_builder(request.builder_id)
            try:
                configuration = spec.resolve_configuration(configuration).model_dump(
                    mode="json"
                )
            except InvalidInvocation as error:
                raise InvalidScheduleRequestError(str(error)) from error
            # A preprocessing app, for example, gets only a slide.
            names = []
            for name, input_spec in spec.inputs.items():
                try:
                    if input_spec.required or await gateway.has_input(name):
                        names.append(name)
                except EmpaiaGatewayError as error:
                    _raise_preparation_error(error)
        effective_deadline = min(deadline_at, credential_expires_at)
        invocation = Invocation(
            inputs={name: ProviderInput(name=name) for name in names},
            resolved_configuration=deepcopy(configuration),
            input_access=InputAccessContext(
                kind=InputAccessKind.EMPAIA,
                state=EmpaiaInputState(
                    api_url=context.api_url, job_id=context.job_id, mode=mode
                ).model_dump(mode="json"),
            ),
            deadline_at=effective_deadline,
        )
        try:
            validate_invocation(
                spec,
                invocation,
                now=effective_deadline - timedelta(microseconds=1),
            )
        except (TypeError, ValueError) as error:
            raise InvalidScheduleRequestError(str(error)) from error
        return PreparedInvocation(
            invocation=invocation,
            protected_context=context.protected_values(),
            deadline_cap=credential_expires_at,
            terminal_report_deadline_at=credential_expires_at,
            # A standalone job was started by hand and someone waits for it;
            # preprocessing runs unattended.
            priority=(
                Priority.HIGH if mode is EmpaiaJobMode.STANDALONE else Priority.NORMAL
            ),
        )

    async def finish(
        self,
        protected_context: dict[str, JsonValue],
        outcome: ProviderFinishOutcome,
        *,
        before_mutation: BeforeMutation | None = None,
    ) -> None:
        """Publish stored outputs and finalize, or report failure/cancellation."""
        context = _job_context(protected_context)
        if not isinstance(outcome, ProviderSuccess):
            # A failure or cancellation report needs nothing from the workspace.
            await self._remove_workspace(context)
        async with _finish_gateway(
            self._gateway_factory,
            context,
        ) as gateway:
            if before_mutation is not None:
                gateway.set_before_mutation(before_mutation)
            mutation_completed = False
            try:
                if isinstance(outcome, ProviderSuccess):
                    for name, output in outcome.result.outputs.items():
                        await gateway.publish_output(name, output)
                        mutation_completed = True
                    await gateway.finalize()
                elif isinstance(outcome, ProviderFailure):
                    await gateway.report_failure(outcome.error.message)
                elif isinstance(outcome, ProviderCancellation):
                    await gateway.report_failure("execution was cancelled")
            except EmpaiaGatewayError as error:
                raise ProviderFinishFailureError(
                    code=EmpaiaErrorCode.FINISH_FAILED,
                    safe_message=error.safe_message,
                    retryable=error.retryable,
                    outcome_unknown=error.outcome_unknown or mutation_completed,
                ) from error
        if isinstance(outcome, ProviderSuccess):
            # Only after publication: a failed one still needs the outputs to retry.
            await self._remove_workspace(context)

    async def _remove_workspace(self, context: EmpaiaJobContext) -> None:
        """Remove every attempt's inputs and outputs of the job."""
        if self._workspace_root is None:
            return
        directory = job_workspace(self._workspace_root, context.api_url, context.job_id)
        await asyncio.to_thread(shutil.rmtree, directory, True)


@asynccontextmanager
async def _preparation_gateway(
    factory: EmpaiaGatewayFactory,
    context: EmpaiaJobContext,
) -> AsyncIterator[EmpaiaGateway]:
    """Own a preparation gateway and translate acquisition/cleanup failures without replacing a primary preparation error."""
    try:
        gateway = factory(context)
    except EmpaiaGatewayError as error:
        _raise_preparation_error(error)
    except Exception as error:
        raise SchedulingUnavailableError(
            "EMPAIA integration is temporarily unavailable"
        ) from error
    try:
        yield gateway
    except BaseException:
        await _close_preserving_outcome(gateway)
        raise
    try:
        await gateway.close()
    except EmpaiaGatewayError as error:
        _raise_preparation_error(error)
    except Exception as error:
        raise SchedulingUnavailableError(
            "EMPAIA integration is temporarily unavailable"
        ) from error


@asynccontextmanager
async def _finish_gateway(
    factory: EmpaiaGatewayFactory,
    context: EmpaiaJobContext,
) -> AsyncIterator[EmpaiaGateway]:
    """Own a finish gateway, translate acquisition failures, and preserve the external completion outcome if cleanup fails."""
    try:
        gateway = factory(context)
    except EmpaiaGatewayError as error:
        raise _finish_failure(error) from error
    except Exception as error:
        raise ProviderFinishFailureError(
            code=EmpaiaErrorCode.FINISH_FAILED,
            safe_message="EMPAIA integration is temporarily unavailable",
            retryable=True,
        ) from error
    try:
        yield gateway
    except BaseException:
        await _close_preserving_outcome(gateway)
        raise
    try:
        await gateway.close()
    except Exception as error:
        logger.warning("EMPAIA gateway cleanup failed: %s", type(error).__name__)


async def _close_preserving_outcome(gateway: EmpaiaGateway) -> None:
    """Close a gateway and log cleanup errors without replacing an already established outcome."""
    try:
        await gateway.close()
    except Exception as error:
        logger.warning("EMPAIA gateway cleanup failed: %s", type(error).__name__)


def _raise_preparation_error(error: EmpaiaGatewayError) -> Never:
    """Map retryable gateway errors to scheduling unavailability and other errors to invalid requests."""
    if error.retryable:
        raise SchedulingUnavailableError(error.safe_message) from error
    raise InvalidScheduleRequestError(error.safe_message) from error


def _finish_failure(error: EmpaiaGatewayError) -> ProviderFinishFailureError:
    """Translate safe gateway retry and uncertainty metadata into a provider finish failure."""
    return ProviderFinishFailureError(
        code=EmpaiaErrorCode.FINISH_FAILED,
        safe_message=error.safe_message,
        retryable=error.retryable,
        outcome_unknown=error.outcome_unknown,
    )


def _allowed(api_url: str, allowed: tuple[str, ...]) -> bool:
    url = api_url.rstrip("/")
    return any(url == prefix or url.startswith(prefix + "/") for prefix in allowed)


def _schedule_payload(values: dict[str, JsonValue]) -> EmpaiaSchedulePayload:
    """Validate provider admission fields, hiding detailed credential-bearing validation input from public errors."""
    try:
        return EmpaiaSchedulePayload.model_validate(values)
    except ValidationError as error:
        raise InvalidScheduleRequestError(
            "EMPAIA schedule payload is invalid"
        ) from error


def _job_context(values: dict[str, JsonValue]) -> EmpaiaJobContext:
    """Validate stored protected job context and return a safe error if it is malformed."""
    try:
        return EmpaiaJobContext.model_validate(values)
    except ValidationError as error:
        raise InvalidScheduleRequestError(
            "EMPAIA protected job context is invalid"
        ) from error


def _credential_expiry(credential: str) -> datetime:
    """Read a numeric exp claim from the credential to cap deadlines. This parses the JWT payload without verifying its signature."""
    try:
        claims = jwt.decode(
            credential,
            options={
                "verify_signature": False,
                "verify_exp": False,
                "require": ["exp"],
            },
        )
        expires_at = claims["exp"]
        if (
            not isinstance(expires_at, int | float)
            or isinstance(expires_at, bool)
            or not math.isfinite(expires_at)
        ):
            raise TypeError
        return datetime.fromtimestamp(expires_at, tz=UTC)
    except (
        jwt.InvalidTokenError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        OSError,
    ) as error:
        raise InvalidScheduleRequestError(
            "EMPAIA credential does not contain a valid expiry"
        ) from error
