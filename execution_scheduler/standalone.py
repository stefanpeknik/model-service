from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from execution_scheduler.contracts import (
    InvalidScheduleRequestError,
    ScheduleRequest,
)
from execution_scheduler.domain import PreparedWorkflow, SubmissionIdentity
from workflow_execution.application import validate_workflow_invocation
from workflow_execution.contracts import SharedPathInput, WorkflowExecutionContext
from workflow_execution.serialization import invocation_from_dict
from workflow_execution.shared_storage import SharedStorageAccessAdapter


if TYPE_CHECKING:
    from execution_scheduler.providers import BeforeMutation, ProviderFinishOutcome
    from workflow_execution.catalog import WorkflowCatalog
    from workflow_execution.contracts import JsonValue


class StandaloneProviderAdapter:
    name = "standalone"

    def __init__(self, catalog: WorkflowCatalog) -> None:
        self._catalog = catalog

    def identify(self, request: ScheduleRequest) -> SubmissionIdentity:
        if not request.idempotency_key:
            raise InvalidScheduleRequestError(
                "standalone scheduling requires an idempotency key"
            )
        try:
            canonical = json.dumps(
                {
                    "workflow_id": request.workflow_id,
                    "body": request.body,
                    "requested_timeout_seconds": request.requested_timeout_seconds,
                },
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        except (TypeError, ValueError) as error:
            raise InvalidScheduleRequestError(
                "standalone schedule request must contain JSON-compatible values"
            ) from error
        return SubmissionIdentity(
            provider=self.name,
            key=request.idempotency_key,
            fingerprint=hashlib.sha256(canonical).hexdigest(),
        )

    async def prepare(
        self,
        request: ScheduleRequest,
        deadline_at: datetime,
    ) -> PreparedWorkflow:
        try:
            invocation = invocation_from_dict(
                {**request.body, "deadline_at": deadline_at.isoformat()}
            )
            definition = self._catalog.get(request.workflow_id)
            self._catalog.validate_configuration(
                request.workflow_id,
                invocation.resolved_configuration,
            )
            validate_workflow_invocation(
                definition,
                invocation,
                now=deadline_at - timedelta(microseconds=1),
            )
            if invocation.input_access.kind != "shared_storage":
                raise ValueError(
                    "standalone scheduling requires shared-storage input access"
                )
            access = SharedStorageAccessAdapter().create(
                invocation, WorkflowExecutionContext()
            )
            for name, value in invocation.inputs.items():
                if isinstance(value, SharedPathInput):
                    access.input_path(name)
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidScheduleRequestError(str(error)) from error
        return PreparedWorkflow(invocation=invocation, protected_context={})

    async def finish(
        self,
        _protected_context: dict[str, JsonValue],
        _outcome: ProviderFinishOutcome,
        *,
        before_mutation: BeforeMutation,
    ) -> None:
        del before_mutation
