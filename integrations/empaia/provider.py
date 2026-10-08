"""Translate EMPAIA jobs into durable executions and publish their stored outcomes.

Admission fetches configuration and caps work/reporting deadlines at credential expiry.
Finish retries are separate from computation and stop when publication's external
outcome is uncertain.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Never

import jwt
from pydantic import Field, ValidationError

from execution_scheduler.contracts import (
    InvalidScheduleRequestError,
    ScheduleRequest,
    SchedulingUnavailableError,
)
from execution_scheduler.domain import PreparedWorkflow, SubmissionIdentity
from execution_scheduler.providers import (
    ProviderCancellation,
    ProviderFailure,
    ProviderFinishFailureError,
    ProviderFinishOutcome,
    ProviderSuccess,
)
from integrations.empaia.gateway import (
    EmpaiaGateway,
    EmpaiaGatewayError,
    EmpaiaGatewayFactory,
    EmpaiaInputState,
    EmpaiaJobContext,
)
from workflow_execution.application import validate_workflow_invocation
from workflow_execution.contracts import (
    InputAccessContext,
    ProviderInput,
    WorkflowInvocation,
    is_json_value,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from execution_scheduler.providers import BeforeMutation
    from workflow_execution.catalog import WorkflowCatalog
    from workflow_execution.contracts import JsonValue


class EmpaiaErrorCode(StrEnum):
    """Name the safe failure for unsuccessful EMPAIA publication or terminal reporting."""

    FINISH_FAILED = "empaia_finish_failed"


class EmpaiaSchedulePayload(EmpaiaJobContext):
    """Require an installation ID alongside the job origin, ID, and private credential.

    Attributes:
        installation_id: Provider installation namespace used to scope job replay
            identity.
    """

    installation_id: Annotated[str, Field(min_length=1)]


logger = logging.getLogger(__name__)


class EmpaiaProviderAdapter:
    """Implement provider-scoped admission and terminal publication for EMPAIA jobs."""

    name = "empaia"

    def __init__(
        self,
        catalog: WorkflowCatalog,
        gateway_factory: EmpaiaGatewayFactory,
    ) -> None:
        """Retain the catalog and gateway factory used for configuration lookup and external job completion.

        Args:
            catalog: Workflow definitions and their configuration validators/defaults.
            gateway_factory: Callable constructing a job gateway from protected
                connection context.
        """
        self._catalog = catalog
        self._gateway_factory = gateway_factory

    def identify(self, request: ScheduleRequest) -> SubmissionIdentity:
        """Key admission by installation and job, fingerprinting stable request fields while excluding renewable credentials.

        Args:
            request: Provider submission with workflow ID, caller body, replay key, and
                optional execution budget.

        Returns:
            SubmissionIdentity: Provider-scoped replay key and canonical request
                fingerprint.

        Raises:
            InvalidScheduleRequestError: Provider payload, configuration, paths, or
                effective deadline are invalid.
        """
        context = _schedule_payload(request.body)
        canonical = json.dumps(
            {
                "workflow_id": request.workflow_id,
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
    ) -> PreparedWorkflow:
        """Fetch and resolve configuration, create provider input references, and separate public job state from protected credentials. Cap deadlines at credential expiry.

        Args:
            request: Provider submission with workflow ID, caller body, replay key, and
                optional execution budget.
            deadline_at: Absolute aware deadline for the entire execution, including
                preparation.

        Returns:
            PreparedWorkflow: Validated invocation, private provider state, and
                applicable deadline limits.

        Raises:
            InvalidScheduleRequestError: Provider payload, configuration, paths, or
                effective deadline are invalid.
            SchedulingUnavailableError: Admission or durable storage is temporarily
                unavailable.
        """
        context = _schedule_payload(request.body)
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
                configuration = self._catalog.resolve_configuration(
                    request.workflow_id,
                    configuration,
                )
            except (TypeError, ValueError) as error:
                raise InvalidScheduleRequestError(str(error)) from error
        effective_deadline = min(deadline_at, credential_expires_at)
        definition = self._catalog.get(request.workflow_id)
        invocation = WorkflowInvocation(
            inputs={name: ProviderInput(name=name) for name in definition.inputs},
            resolved_configuration=deepcopy(configuration),
            input_access=InputAccessContext(
                kind="empaia",
                state=EmpaiaInputState(
                    api_url=context.api_url, job_id=context.job_id
                ).model_dump(mode="json"),
            ),
            deadline_at=effective_deadline,
        )
        try:
            validate_workflow_invocation(
                definition,
                invocation,
                now=effective_deadline - timedelta(microseconds=1),
            )
        except (TypeError, ValueError) as error:
            raise InvalidScheduleRequestError(str(error)) from error
        return PreparedWorkflow(
            invocation=invocation,
            protected_context=context.protected_values(),
            deadline_cap=credential_expires_at,
            terminal_report_deadline_at=credential_expires_at,
        )

    async def finish(
        self,
        protected_context: dict[str, JsonValue],
        outcome: ProviderFinishOutcome,
        *,
        before_mutation: BeforeMutation | None = None,
    ) -> None:
        """Publish stored outputs and finalize, or report failure/cancellation.

        Install the durable mutation guard when supplied. A failure after an output was
        published is outcome-unknown and cannot safely trigger blind publication
        retries.

        Args:
            protected_context: Private provider state that persistence encrypts and
                internal dispatch carries separately.
            outcome: Stored success, safe failure, or cancellation to report through the
                provider.
            before_mutation: Async guard awaited before the first external write to
                persist and validate its durable claim.

        Returns:
            None: No return value.

        Raises:
            InvalidScheduleRequestError: Provider payload, configuration, paths, or
                effective deadline are invalid.
            ProviderFinishFailureError: Publication or terminal reporting fails;
                metadata states whether retry is safe.
            StaleWorkClaimError: The token, lease, deadline, or work kind no longer
                permits this operation.
        """
        context = _job_context(protected_context)
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
                    await gateway.report_failure("workflow execution was cancelled")
            except EmpaiaGatewayError as error:
                raise ProviderFinishFailureError(
                    code=EmpaiaErrorCode.FINISH_FAILED,
                    safe_message=error.safe_message,
                    retryable=error.retryable,
                    outcome_unknown=error.outcome_unknown or mutation_completed,
                ) from error


@asynccontextmanager
async def _preparation_gateway(
    factory: EmpaiaGatewayFactory,
    context: EmpaiaJobContext,
) -> AsyncIterator[EmpaiaGateway]:
    """Own a preparation gateway and translate acquisition/cleanup failures without replacing a primary preparation error.

    Args:
        factory: Callable that creates a provider gateway for one protected job context.
        context: External job origin/ID and protected bearer credential.

    Yields:
        EmpaiaGateway: Job-scoped gateway that is closed when the context exits.

    Raises:
        SchedulingUnavailableError: Admission or durable storage is temporarily
            unavailable.
    """
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
    """Own a finish gateway, translate acquisition failures, and preserve the external completion outcome if cleanup fails.

    Args:
        factory: Callable that creates a provider gateway for one protected job context.
        context: External job origin/ID and protected bearer credential.

    Yields:
        EmpaiaGateway: Job-scoped gateway that is closed when the context exits.

    Raises:
        ProviderFinishFailureError: Publication or terminal reporting fails; metadata
            states whether retry is safe.
    """
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
    """Close a gateway and log cleanup errors without replacing an already established outcome.

    Args:
        gateway: Job-scoped provider boundary used for publication or cleanup.

    Returns:
        None: No return value.
    """
    try:
        await gateway.close()
    except Exception as error:
        logger.warning("EMPAIA gateway cleanup failed: %s", type(error).__name__)


def _raise_preparation_error(error: EmpaiaGatewayError) -> Never:
    """Map retryable gateway errors to scheduling unavailability and other errors to invalid requests.

    Args:
        error: Safe domain error or injected integration exception to classify.

    Raises:
        InvalidScheduleRequestError: Provider payload, configuration, paths, or
            effective deadline are invalid.
        SchedulingUnavailableError: Admission or durable storage is temporarily
            unavailable.
    """
    if error.retryable:
        raise SchedulingUnavailableError(error.safe_message) from error
    raise InvalidScheduleRequestError(error.safe_message) from error


def _finish_failure(error: EmpaiaGatewayError) -> ProviderFinishFailureError:
    """Translate safe gateway retry and uncertainty metadata into a provider finish failure.

    Args:
        error: Safe domain error or injected integration exception to classify.

    Returns:
        ProviderFinishFailureError: Safe provider finish error with retry and
            uncertainty classification.
    """
    return ProviderFinishFailureError(
        code=EmpaiaErrorCode.FINISH_FAILED,
        safe_message=error.safe_message,
        retryable=error.retryable,
        outcome_unknown=error.outcome_unknown,
    )


def _schedule_payload(values: dict[str, JsonValue]) -> EmpaiaSchedulePayload:
    """Validate provider admission fields, hiding detailed credential-bearing validation input from public errors.

    Args:
        values: Configuration/context/environment mapping consumed by this validator.

    Returns:
        EmpaiaSchedulePayload: Validated installation/job admission context.

    Raises:
        InvalidScheduleRequestError: Provider payload, configuration, paths, or
            effective deadline are invalid.
    """
    try:
        return EmpaiaSchedulePayload.model_validate(values)
    except ValidationError as error:
        raise InvalidScheduleRequestError(
            "EMPAIA schedule payload is invalid"
        ) from error


def _job_context(values: dict[str, JsonValue]) -> EmpaiaJobContext:
    """Validate stored protected job context and return a safe error if it is malformed.

    Args:
        values: Configuration/context/environment mapping consumed by this validator.

    Returns:
        EmpaiaJobContext: Validated job origin, identity, and private bearer credential.

    Raises:
        InvalidScheduleRequestError: Provider payload, configuration, paths, or
            effective deadline are invalid.
    """
    try:
        return EmpaiaJobContext.model_validate(values)
    except ValidationError as error:
        raise InvalidScheduleRequestError(
            "EMPAIA protected job context is invalid"
        ) from error


def _credential_expiry(credential: str) -> datetime:
    """Read a numeric exp claim from the credential to cap deadlines. This parses the JWT payload without verifying its signature.

    Args:
        credential: Private bearer credential for the external job; never include it in
            public errors.

    Returns:
        datetime: Credential expiration time in UTC, read from its numeric exp claim.

    Raises:
        InvalidScheduleRequestError: Provider payload, configuration, paths, or
            effective deadline are invalid.
    """
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
