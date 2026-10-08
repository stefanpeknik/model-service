"""Prepare SDK/shared-storage executions without an external job lifecycle.

Caller replay keys provide identity. Finish is a no-op because outputs already reside in
shared storage, but it still passes through the common scheduler finish phase.
"""

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
from integrations.shared_storage import SharedStorageAccessAdapter
from workflow_execution.application import validate_workflow_invocation
from workflow_execution.contracts import SharedPathInput, WorkflowExecutionContext
from workflow_execution.serialization import invocation_from_dict


if TYPE_CHECKING:
    from execution_scheduler.providers import BeforeMutation, ProviderFinishOutcome
    from workflow_execution.catalog import WorkflowCatalog
    from workflow_execution.contracts import JsonValue


class StandaloneProviderAdapter:
    """Adapt shared-storage schedule requests to the scheduler's provider interface."""

    name = "standalone"

    def __init__(self, catalog: WorkflowCatalog) -> None:
        """Retain the workflow catalog used to validate configuration and named inputs.

        Args:
            catalog: Workflow definitions and their configuration validators/defaults.
        """
        self._catalog = catalog

    def identify(self, request: ScheduleRequest) -> SubmissionIdentity:
        """Require a replay key and fingerprint canonical finite JSON including workflow, body, and requested timeout.

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
        """Build a typed invocation with the scheduler deadline and validate configuration, input kinds, and shared input file availability.

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
        """
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
        """Complete without an external mutation because standalone results already reference shared artifacts.

        Args:
            _protected_context: Unused private provider state that persistence encrypts
                and internal dispatch carries separately; retained to match the
                interface.
            _outcome: Unused stored success, safe failure, or cancellation to report
                through the provider; retained to match the interface.
            before_mutation: Async guard awaited before the first external write to
                persist and validate its durable claim.

        Returns:
            None: No return value.
        """
        del before_mutation
