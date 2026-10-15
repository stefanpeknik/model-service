"""Standalone provider: shared-storage inputs and outputs, no external job lifecycle.

The caller's idempotency key identifies a submission. Finish is a no-op because
outputs already sit in shared storage, but it still runs through the common
provider-finish phase.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import ValidationError

from model_service.builders.application import validate_invocation
from model_service.builders.contracts import (
    ExecutionContext,
    InvalidInvocation,
    SharedPathInput,
)
from model_service.builders.registry import get_builder
from model_service.facade.requests import BuilderRequest
from model_service.scheduler.contracts import (
    InvalidScheduleRequestError,
    ScheduleRequest,
)
from model_service.scheduler.domain import PreparedInvocation, SubmissionIdentity


if TYPE_CHECKING:
    from model_service.builders.contracts import JsonValue
    from model_service.providers.shared_storage import SharedStorageAccessAdapter
    from model_service.scheduler.providers import BeforeMutation, ProviderFinishOutcome


class StandaloneProviderAdapter:
    name = "standalone"

    def __init__(self, shared_storage: SharedStorageAccessAdapter) -> None:
        self._shared_storage = shared_storage

    def identify(self, request: ScheduleRequest) -> SubmissionIdentity:
        if not request.idempotency_key:
            raise InvalidScheduleRequestError(
                "standalone scheduling requires an idempotency key"
            )
        try:
            canonical = json.dumps(
                {
                    "builder_id": request.builder_id,
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
        self, request: ScheduleRequest, deadline_at: datetime
    ) -> PreparedInvocation:
        """Check inputs exist now; processing settings are chosen per attempt."""
        try:
            body = BuilderRequest.model_validate(request.body)
        except ValidationError as error:
            raise InvalidScheduleRequestError(
                "schedule request body is invalid"
            ) from error
        if body.model_id != request.model_id:
            raise InvalidScheduleRequestError("model_id does not match the request")
        invocation = body.invocation(deadline_at)
        try:
            validate_invocation(
                get_builder(request.builder_id),
                invocation,
                now=deadline_at - timedelta(microseconds=1),
            )
            if invocation.input_access.kind != "shared_storage":
                raise InvalidInvocation("standalone scheduling requires shared storage")
            access = self._shared_storage.create(invocation, ExecutionContext())
            for name, value in invocation.inputs.items():
                if isinstance(value, SharedPathInput):
                    access.input_path(name)
        except InvalidInvocation as error:
            raise InvalidScheduleRequestError(str(error)) from error
        return PreparedInvocation(invocation=invocation, protected_context={})

    async def finish(
        self,
        _protected_context: dict[str, JsonValue],
        _outcome: ProviderFinishOutcome,
        *,
        before_mutation: BeforeMutation,
    ) -> None:
        del before_mutation
