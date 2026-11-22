"""Standalone provider, API side: admission and finish of shared-storage jobs.

The caller's idempotency key identifies a submission. Scheduled attempts write
below `.attempts/<execution>/` in the shared-storage root; finish moves the
accepted attempt's artifacts to their requested paths and removes that
directory, including failed and superseded attempts. Only the API process
imports this module; the analysis runner loads provider.py alone.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import ValidationError

from model_service.api.scheduler.public import (
    AnalysisScheduleRequest,
    InvalidScheduleRequestError,
    PreparedInvocation,
    ProviderFinishFailureError,
    ProviderSuccess,
    SubmissionIdentity,
)
from model_service.contracts.caller_request import BuilderRequest
from model_service.contracts.failures import InvalidInvocation
from model_service.contracts.invocation import (
    SHARED_STORAGE,
    ExecutionContext,
    SharedPathInput,
)
from model_service.contracts.result import BuilderResult, SharedArtifact
from model_service.directories import make_shared_directory, remove_directory
from model_service.providers.provider import ProviderAdapter
from model_service.providers.standalone.provider import STANDALONE_PROVIDER
from model_service.providers.standalone.shared_storage import (
    ATTEMPTS_DIR,
    ATTEMPTS_KEY,
    SharedStorageAccess,
    SharedStorageAccessFactory,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.api.scheduler.public import (
        BeforeMutation,
        ProviderFinishOutcome,
    )
    from model_service.contracts.json_values import JsonValue
    from model_service.contracts.result import BuilderOutput


class StandaloneProviderAdapter(ProviderAdapter):
    """Scheduler adapter for runs whose inputs and outputs are on shared storage."""

    name = STANDALONE_PROVIDER

    def __init__(self, shared_storage: SharedStorageAccessFactory) -> None:
        """Keep the shared-storage access that checks requested roots.

        Args:
            shared_storage: The shared-storage access factory.
        """
        self._shared_storage = shared_storage

    def identify(self, request: AnalysisScheduleRequest) -> SubmissionIdentity:
        """Identify a submission by its idempotency key and a fingerprint of the request.

        The fingerprint hashes the canonical JSON of the analysis, body and
        requested timeout, so reusing a key for a different request is
        detected.

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
                    "analysis_id": request.analysis_id,
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
        self, request: AnalysisScheduleRequest, deadline_at: datetime
    ) -> PreparedInvocation:
        """Check the request's access and shared inputs.

        The invocation keeps the caller's settings: the scheduler resolves
        the analysis with them and checks the inputs against it.

        Args:
            request: The schedule request; its body is a `BuilderRequest`.
            deadline_at: When the execution must finish; validation runs as of just
                before it.

        Returns:
            The invocation, naming every setting it runs with, and the root and
            a fresh attempts directory name as protected context.

        Raises:
            InvalidScheduleRequestError: If the body is invalid, input access is
                not shared storage, or a shared input is not a regular file below
                an allowed root.
        """
        try:
            body = BuilderRequest.model_validate(request.body)
        except ValidationError as error:
            raise InvalidScheduleRequestError(
                "schedule request body is invalid"
            ) from error
        invocation = body.invocation(deadline_at)
        try:
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
            try:
                outputs = await asyncio.to_thread(
                    _move_outputs, Path(root), attempts, outcome.result.outputs
                )
            except (OSError, InvalidInvocation) as error:
                raise ProviderFinishFailureError(
                    code="output_unavailable",
                    safe_message="could not move the result to its output path",
                    retryable=True,
                ) from error
            result = BuilderResult(outputs)
        await asyncio.to_thread(remove_directory, attempts)
        return result


def _move_outputs(
    root: Path, attempts: Path, outputs: Mapping[str, BuilderOutput]
) -> dict[str, BuilderOutput]:
    """Move an attempt's artifacts to their requested paths under the root.

    Blocking file work, so the caller runs it in a thread.

    Args:
        root: The caller's shared storage root.
        attempts: The execution's attempts directory.
        outputs: The run's outputs.

    Returns:
        The outputs, with moved artifacts at their requested paths.

    Raises:
        OSError: A file could not be moved.
        InvalidInvocation: A requested path leaves the root.
    """
    moved = dict(outputs)
    storage = SharedStorageAccess(root, {})
    for name, output in outputs.items():
        if not isinstance(output, SharedArtifact):
            continue
        source = Path(output.path)
        if not source.is_relative_to(attempts):
            continue
        # Drop the attempt's own directory from the relative path.
        relative = source.relative_to(attempts).parts[1:]
        destination = storage.output_path("/".join(relative))
        if source.exists():
            make_shared_directory(destination.parent)
            os.replace(source, destination)
        moved[name] = replace(output, path=str(destination))
    return moved
