"""Standalone provider: shared-storage inputs and outputs, no external job lifecycle.

The caller's idempotency key identifies a submission. Finish is a no-op because
outputs already sit in shared storage, but it still runs through the common
provider finish.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from model_service.builders.contracts import (
    SHARED_STORAGE,
    BuilderInput,
    Deadline,
    ExecutionContext,
    InputAccessContext,
    InvalidInvocation,
    Invocation,
    SharedPathInput,
)
from model_service.builders.registry import get_builder
from model_service.providers.composition import Provider
from model_service.scheduler.contracts import (
    InvalidScheduleRequestError,
    ProviderName,
    ScheduleRequest,
)
from model_service.scheduler.domain import PreparedInvocation, SubmissionIdentity
from model_service.scheduler.providers import ProviderAdapter


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.builders.access import AccessAdapter
    from model_service.builders.contracts import JsonValue
    from model_service.builders.settings import WorkerSettings
    from model_service.facade.catalog import ModelCatalog
    from model_service.facade.settings import SchedulerSettings
    from model_service.providers.composition import FacadeContribution
    from model_service.providers.shared_storage import SharedStorageAccessAdapter
    from model_service.scheduler.providers import BeforeMutation, ProviderFinishOutcome


NAME: ProviderName = "standalone"


class BuilderRequest(BaseModel):
    """Inputs and their access, and an optional output path.

    Processing settings are not part of the request; each attempt uses the
    builder's current defaults.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    inputs: Annotated[dict[str, BuilderInput], Field(strict=True)]
    input_access: InputAccessContext
    output_path: Annotated[str, Field(strict=True, min_length=1)] | None = None

    def invocation(self, deadline_at: Deadline) -> Invocation:
        """Build the invocation of this request; the attempt resolves the configuration.

        Args:
            deadline_at: When the execution must finish.

        Returns:
            The invocation, with no resolved configuration.
        """
        return Invocation(
            inputs=self.inputs,
            resolved_configuration=None,
            input_access=self.input_access,
            deadline_at=deadline_at,
            output_path=self.output_path,
        )


class StandaloneProviderAdapter(ProviderAdapter):
    name = NAME

    def __init__(self, shared_storage: SharedStorageAccessAdapter) -> None:
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
    """Shared-storage jobs, through the public builder routes the facade serves."""

    catalog_block = None

    def facade(
        self, catalog: ModelCatalog, settings: SchedulerSettings
    ) -> FacadeContribution:
        """Contribute the standalone scheduler adapter, with no routes or background work.

        The facade's public builder routes submit these jobs.

        Args:
            catalog: Unused.
            settings: Supplies the allowed shared-storage roots.

        Returns:
            The facade contribution.
        """
        from model_service.providers.composition import FacadeContribution
        from model_service.providers.shared_storage import SharedStorageAccessAdapter

        del catalog
        return FacadeContribution(
            StandaloneProviderAdapter(
                SharedStorageAccessAdapter(settings.shared_storage_roots)
            )
        )

    def input_access(self, settings: WorkerSettings) -> Mapping[str, AccessAdapter]:
        """Give builder replicas shared-storage access below the allowed roots.

        Args:
            settings: Supplies the allowed shared-storage roots.

        Returns:
            The shared-storage adapter, keyed by `SHARED_STORAGE`.
        """
        from model_service.providers.shared_storage import SharedStorageAccessAdapter

        return {
            SHARED_STORAGE: SharedStorageAccessAdapter(settings.shared_storage_roots)
        }


PROVIDER = StandaloneProvider()
