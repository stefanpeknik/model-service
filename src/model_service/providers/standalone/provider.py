"""Standalone provider: shared-storage inputs and outputs, no external job lifecycle.

The caller's idempotency key identifies a submission. Finish is a no-op because
outputs already sit in shared storage, but it still runs through the common
provider finish.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import ValidationError

from model_service.builders.registry import get_builder
from model_service.providers.provider import Provider, ProviderAdapter
from model_service.scheduler.public import (
    InvalidScheduleRequestError,
    PreparedInvocation,
    ScheduleRequest,
    SubmissionIdentity,
)
from model_service.shared.caller_request import STANDALONE_PROVIDER, BuilderRequest
from model_service.shared.failures import InvalidInvocation
from model_service.shared.invocation import (
    SHARED_STORAGE,
    ExecutionContext,
    SharedPathInput,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.builders.base import ReplicaSettings
    from model_service.builders.input_access import InputAccessFactory
    from model_service.inference.catalog import ModelCatalog
    from model_service.providers.provider import ApiContribution
    from model_service.providers.standalone.shared_storage import (
        SharedStorageAccessFactory,
    )
    from model_service.scheduler.public import BeforeMutation, ProviderFinishOutcome
    from model_service.scheduler.settings import SchedulerSettings
    from model_service.shared.json_values import JsonValue


class StandaloneProviderAdapter(ProviderAdapter):
    name = STANDALONE_PROVIDER

    def __init__(self, shared_storage: SharedStorageAccessFactory) -> None:
        """Keep the shared-storage adapter that checks inputs at admission.

        Args:
            shared_storage: The adapter for the allowed shared-storage roots.
        """
        self._shared_storage = shared_storage

    def identify(self, request: ScheduleRequest) -> SubmissionIdentity:
        """Identify a submission by its idempotency key and a fingerprint of the request.

        The fingerprint hashes the canonical JSON of builder, model, body and
        requested timeout, so reusing a key for a different request is detected.

        Args:
            request: The schedule request.

        Returns:
            The submission identity.

        Raises:
            InvalidScheduleRequestError: If the idempotency key is missing or the
                request holds values JSON cannot encode.
        """
        if not request.idempotency_key:
            raise InvalidScheduleRequestError(
                "standalone scheduling requires an idempotency key"
            )
        try:
            canonical = json.dumps(
                {
                    "builder_id": request.builder_id,
                    "model_id": request.model_id,
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
        """Check inputs exist now; processing settings are chosen per attempt.

        Args:
            request: The schedule request; its body is a `BuilderRequest`.
            deadline_at: When the execution must finish; validation runs as of just
                before it.

        Returns:
            The invocation, with no protected context.

        Raises:
            InvalidScheduleRequestError: If the body is invalid, the invocation fails
                the builder's validation, input access is not shared storage, or a
                shared input is not a regular file below an allowed root.
        """
        try:
            body = BuilderRequest.model_validate(request.body)
        except ValidationError as error:
            raise InvalidScheduleRequestError(
                "schedule request body is invalid"
            ) from error
        invocation = body.invocation(deadline_at)
        try:
            get_builder(request.builder_id).check_invocation(
                invocation,
                now=deadline_at - timedelta(microseconds=1),
            )
            if invocation.input_access.kind != SHARED_STORAGE:
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
        """Do nothing; the outputs already sit in shared storage.

        Args:
            _protected_context: Unused; standalone executions have none.
            _outcome: Unused.
            before_mutation: Unused; nothing is written.
        """
        del before_mutation


class StandaloneProvider(Provider):
    """Shared-storage jobs, through the public builder routes the API serves."""

    catalog_block = None

    def api(
        self, catalog: ModelCatalog, settings: SchedulerSettings
    ) -> ApiContribution:
        """Contribute the standalone scheduler adapter, with no routes or background work.

        the API's public builder routes submit these jobs.

        Args:
            catalog: Unused.
            settings: Supplies the allowed shared-storage roots.

        Returns:
            the API contribution.
        """
        from model_service.providers.provider import ApiContribution
        from model_service.providers.standalone.shared_storage import (
            SharedStorageAccessFactory,
        )

        del catalog
        return ApiContribution(
            StandaloneProviderAdapter(
                SharedStorageAccessFactory(settings.shared_storage_roots)
            )
        )

    def input_access(
        self, settings: ReplicaSettings
    ) -> Mapping[str, InputAccessFactory]:
        """Give builder replicas shared-storage access below the allowed roots.

        Args:
            settings: Supplies the allowed shared-storage roots.

        Returns:
            The shared-storage adapter, keyed by `SHARED_STORAGE`.
        """
        from model_service.providers.standalone.shared_storage import (
            SharedStorageAccessFactory,
        )

        return {
            SHARED_STORAGE: SharedStorageAccessFactory(settings.shared_storage_roots)
        }


PROVIDER = StandaloneProvider()
