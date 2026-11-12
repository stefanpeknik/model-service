"""Standalone provider: shared-storage inputs and outputs, no external job lifecycle.

The caller's idempotency key identifies a submission. Scheduled attempts write
below `.attempts/<execution>/` in the shared-storage root; finish moves the
accepted attempt's artifacts to their requested paths and removes that
directory, including failed and superseded attempts.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import ValidationError

from model_service.builders.registry import get_builder
from model_service.contracts.caller_request import BuilderRequest
from model_service.contracts.failures import InvalidInvocation
from model_service.contracts.invocation import (
    SHARED_STORAGE,
    ExecutionContext,
    SharedPathInput,
)
from model_service.contracts.result import BuilderResult, SharedArtifact
from model_service.providers.provider import Provider, ProviderAdapter
from model_service.providers.standalone.shared_storage import (
    ATTEMPTS_DIR,
    ATTEMPTS_KEY,
    SharedStorageAccess,
    SharedStorageAccessFactory,
)
from model_service.scheduler.public import (
    InvalidScheduleRequestError,
    PreparedInvocation,
    ProviderFinishFailureError,
    ProviderSuccess,
    ScheduleRequest,
    SubmissionIdentity,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.builders.base import ReplicaSettings
    from model_service.builders.input_access import InputAccessFactory
    from model_service.contracts.json_values import JsonValue
    from model_service.deployed_models.catalog import ModelCatalog
    from model_service.providers.provider import ApiContribution
    from model_service.scheduler.public import BeforeMutation, ProviderFinishOutcome
    from model_service.scheduler.settings import SchedulerSettings


# The provider name of requests made through the public builder routes.
STANDALONE_PROVIDER = "standalone"


class StandaloneProviderAdapter(ProviderAdapter):
    name = STANDALONE_PROVIDER

    def __init__(self, shared_storage: SharedStorageAccessFactory) -> None:
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
            The invocation, with the root and a fresh attempts directory name as
            protected context.

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
            root = self._shared_storage.root(invocation.input_access.state)
        except InvalidInvocation as error:
            raise InvalidScheduleRequestError(str(error)) from error
        return PreparedInvocation(
            invocation=invocation,
            protected_context={"root": str(root), ATTEMPTS_KEY: uuid4().hex},
        )

    async def finish(
        self,
        protected_context: dict[str, JsonValue],
        outcome: ProviderFinishOutcome,
        *,
        before_mutation: BeforeMutation,
    ) -> BuilderResult | None:
        """Move accepted artifacts to their requested paths and remove the attempts.

        A retry after a partial move finds the moved files in place. Executions
        admitted without an attempts directory keep their outputs where they are.

        Returns:
            On success, the result with the requested paths; otherwise None.

        Raises:
            ProviderFinishFailureError: An artifact could not be moved; retryable.
        """
        del before_mutation
        root, group = protected_context.get("root"), protected_context.get(ATTEMPTS_KEY)
        if not isinstance(root, str) or not isinstance(group, str):
            return None
        attempts = Path(root, ATTEMPTS_DIR, group)
        result = None
        if isinstance(outcome, ProviderSuccess):
            outputs = dict(outcome.result.outputs)
            try:
                for name, output in outputs.items():
                    if not isinstance(output, SharedArtifact):
                        continue
                    source = Path(output.path)
                    if not source.is_relative_to(attempts):
                        continue
                    # Drop the attempt's own directory from the relative path.
                    relative = source.relative_to(attempts).parts[1:]
                    destination = SharedStorageAccess(Path(root), {}).output_path(
                        "/".join(relative)
                    )
                    if source.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(source, destination)
                    outputs[name] = replace(output, path=str(destination))
            except (OSError, InvalidInvocation) as error:
                raise ProviderFinishFailureError(
                    code="output_unavailable",
                    safe_message="could not move the result to its output path",
                    retryable=True,
                ) from error
            result = BuilderResult(outputs)
        shutil.rmtree(attempts, ignore_errors=True)
        return result


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
        return {
            SHARED_STORAGE: SharedStorageAccessFactory(settings.shared_storage_roots)
        }


PROVIDER = StandaloneProvider()
