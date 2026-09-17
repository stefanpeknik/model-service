from __future__ import annotations

import base64
import hashlib
import importlib
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

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
from workflow_execution.application import validate_workflow_invocation
from workflow_execution.contracts import (
    InputAccessContext,
    ProviderInput,
    WorkflowInvocation,
    is_json_value,
)


if TYPE_CHECKING:
    from workflow_execution.catalog import WorkflowCatalog
    from workflow_execution.contracts import JsonValue, WorkflowOutput


class EmpaiaGatewayError(RuntimeError):
    def __init__(
        self,
        safe_message: str,
        *,
        retryable: bool,
        outcome_unknown: bool = False,
    ) -> None:
        super().__init__(safe_message)
        self.safe_message = safe_message
        self.retryable = retryable
        self.outcome_unknown = outcome_unknown


class EmpaiaGateway(Protocol):
    async def get_configuration(self) -> dict[str, JsonValue]: ...

    async def materialize_input(self, name: str, destination: str) -> None: ...

    async def publish_output(self, name: str, output: WorkflowOutput) -> None: ...

    async def finalize(self) -> None: ...

    async def report_failure(self, message: str) -> None: ...

    async def close(self) -> None: ...


class EmpaiaGatewayFactory(Protocol):
    def __call__(
        self,
        api_url: str,
        job_id: str,
        credential: str,
    ) -> EmpaiaGateway: ...


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
                code="empaia_finish_failed",
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


def load_gateway_factory(import_path: str) -> EmpaiaGatewayFactory:
    module_name, separator, attribute_name = import_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("EMPAIA gateway factory must use module:attribute syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise TypeError("EMPAIA gateway factory is not callable")
    return factory
