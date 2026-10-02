from __future__ import annotations

import base64
import hashlib
import json
import logging
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Never

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
    FINISH_FAILED = "empaia_finish_failed"


class EmpaiaSchedulePayload(EmpaiaJobContext):
    installation_id: Annotated[str, Field(min_length=1)]


logger = logging.getLogger(__name__)


class EmpaiaProviderAdapter:
    name = "empaia"

    def __init__(
        self,
        catalog: WorkflowCatalog,
        gateway_factory: EmpaiaGatewayFactory,
    ) -> None:
        self._catalog = catalog
        self._gateway_factory = gateway_factory

    def identify(self, request: ScheduleRequest) -> SubmissionIdentity:
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
    except Exception as error:  # noqa: BLE001 - Cleanup must preserve the committed outcome.
        logger.warning("EMPAIA gateway cleanup failed: %s", type(error).__name__)


async def _close_preserving_outcome(gateway: EmpaiaGateway) -> None:
    try:
        await gateway.close()
    except Exception as error:  # noqa: BLE001 - Cleanup must preserve the original error.
        logger.warning("EMPAIA gateway cleanup failed: %s", type(error).__name__)


def _raise_preparation_error(error: EmpaiaGatewayError) -> Never:
    if error.retryable:
        raise SchedulingUnavailableError(error.safe_message) from error
    raise InvalidScheduleRequestError(error.safe_message) from error


def _finish_failure(error: EmpaiaGatewayError) -> ProviderFinishFailureError:
    return ProviderFinishFailureError(
        code=EmpaiaErrorCode.FINISH_FAILED,
        safe_message=error.safe_message,
        retryable=error.retryable,
        outcome_unknown=error.outcome_unknown,
    )


def _schedule_payload(values: dict[str, JsonValue]) -> EmpaiaSchedulePayload:
    try:
        return EmpaiaSchedulePayload.model_validate(values)
    except ValidationError as error:
        raise InvalidScheduleRequestError(
            "EMPAIA schedule payload is invalid"
        ) from error


def _job_context(values: dict[str, JsonValue]) -> EmpaiaJobContext:
    try:
        return EmpaiaJobContext.model_validate(values)
    except ValidationError as error:
        raise InvalidScheduleRequestError(
            "EMPAIA protected job context is invalid"
        ) from error


def _credential_expiry(credential: str) -> datetime:
    try:
        _header, payload, _signature = credential.split(".")
        padding = "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload + padding))
        expires_at = claims["exp"]
        if not isinstance(expires_at, int | float) or isinstance(expires_at, bool):
            raise TypeError
        return datetime.fromtimestamp(expires_at, tz=UTC)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise InvalidScheduleRequestError(
            "EMPAIA credential does not contain a valid expiry"
        ) from error
