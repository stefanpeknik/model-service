from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING

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
from integrations.empaia.gateway import EmpaiaGatewayError, EmpaiaGatewayFactory
from workflow_execution.application import validate_workflow_invocation
from workflow_execution.contracts import (
    InputAccessContext,
    ProviderInput,
    WorkflowInvocation,
)
from workflow_execution.serialization import is_json_value


if TYPE_CHECKING:
    from workflow_execution.catalog import WorkflowCatalog
    from workflow_execution.contracts import JsonValue


class EmpaiaErrorCode(StrEnum):
    FINISH_FAILED = "empaia_finish_failed"


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
        installation_id = _required_string(request.body, "installation_id")
        job_id = _required_string(request.body, "job_id")
        api_url = _required_string(request.body, "api_url")
        canonical = json.dumps(
            {
                "workflow_id": request.workflow_id,
                "installation_id": installation_id,
                "job_id": job_id,
                "api_url": api_url,
                "requested_timeout_seconds": request.requested_timeout_seconds,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return SubmissionIdentity(
            provider=self.name,
            key=f"{installation_id}:{job_id}",
            fingerprint=hashlib.sha256(canonical).hexdigest(),
        )

    async def prepare(
        self,
        request: ScheduleRequest,
        deadline_at: datetime,
    ) -> PreparedWorkflow:
        api_url = _required_string(request.body, "api_url")
        job_id = _required_string(request.body, "job_id")
        credential = _required_string(request.body, "credential")
        credential_expires_at = _credential_expiry(credential)
        gateway = self._gateway_factory(api_url, job_id, credential)
        try:
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
                self._catalog.validate_configuration(
                    request.workflow_id,
                    configuration,
                )
            except (TypeError, ValueError) as error:
                raise InvalidScheduleRequestError(str(error)) from error
        finally:
            await gateway.close()
        effective_deadline = min(deadline_at, credential_expires_at)
        definition = self._catalog.get(request.workflow_id)
        invocation = WorkflowInvocation(
            inputs={name: ProviderInput(name=name) for name in definition.inputs},
            resolved_configuration=deepcopy(configuration),
            input_access=InputAccessContext(
                kind="empaia",
                state={"api_url": api_url, "job_id": job_id},
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
            protected_context={
                "api_url": api_url,
                "job_id": job_id,
                "credential": credential,
            },
            deadline_cap=credential_expires_at,
        )

    async def finish(
        self,
        protected_context: dict[str, JsonValue],
        outcome: ProviderFinishOutcome,
    ) -> None:
        api_url = _required_string(protected_context, "api_url")
        job_id = _required_string(protected_context, "job_id")
        credential = _required_string(protected_context, "credential")
        gateway = self._gateway_factory(api_url, job_id, credential)
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
        finally:
            await gateway.close()


def _required_string(values: dict[str, JsonValue], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value:
        raise InvalidScheduleRequestError(f"{name} must be a non-empty string")
    return value


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
