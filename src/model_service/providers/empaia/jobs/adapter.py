"""Translate EMPAIA jobs into durable executions and publish their stored outcomes.

Admission fetches configuration and caps work/reporting deadlines at credential expiry.
Finish retries are separate from computation and stop when publication's external
outcome is uncertain.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
from copy import deepcopy
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import ValidationError

from model_service.builders.registry import get_builder
from model_service.contracts.invocation import (
    InputAccessContext,
    Invocation,
    ProviderInput,
)
from model_service.contracts.json_values import is_json_value
from model_service.providers.empaia.app_service.gateway import (
    INPUT_ACCESS_KIND,
    PROVIDER_NAME,
    EmpaiaGatewayError,
    EmpaiaGatewayFactory,
    EmpaiaInputState,
    EmpaiaJobContext,
    EmpaiaJobMode,
)
from model_service.providers.empaia.ead.builder_io import BUILDER_IO
from model_service.providers.empaia.jobs.gateways import (
    FINISH_FAILED,
    finish_gateway,
    preparation_gateway,
)
from model_service.providers.empaia.jobs.inputs import job_workspace
from model_service.providers.empaia.jobs.mapping import (
    builder_configuration,
    job_inputs,
    raise_preparation_error,
)
from model_service.providers.empaia.jobs.payload import (
    allowed,
    credential_expiry,
    job_context,
    schedule_payload,
)
from model_service.providers.empaia.publishing.publishers import default_publishers
from model_service.providers.provider import ProviderAdapter
from model_service.scheduler.public import (
    InvalidScheduleRequestError,
    PreparedInvocation,
    Priority,
    ProviderCancellation,
    ProviderFailure,
    ProviderFinishFailureError,
    ProviderFinishOutcome,
    ProviderSuccess,
    ScheduleRequest,
    SubmissionIdentity,
)


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from model_service.contracts.json_values import JsonValue
    from model_service.providers.empaia.ead.builder_io import BuilderIo
    from model_service.providers.empaia.publishing.publishers import PublisherRegistry
    from model_service.scheduler.public import BeforeMutation


logger = logging.getLogger(__name__)


class EmpaiaProviderAdapter(ProviderAdapter):
    """Implement provider-scoped admission and terminal publication for EMPAIA jobs."""

    name = PROVIDER_NAME

    def __init__(
        self,
        gateway_factory: EmpaiaGatewayFactory,
        *,
        allowed_api_urls: tuple[str, ...],
        workspace_root: Path | None = None,
        publishers: PublisherRegistry | None = None,
        io: Mapping[str, BuilderIo] = BUILDER_IO,
    ) -> None:
        """Only `allowed_api_urls` origins or path prefixes receive job credentials; none, no job.

        Args:
            gateway_factory: Creates the gateway of each job.
            allowed_api_urls: App Service origins or path prefixes allowed to receive
                job credentials.
            workspace_root: The root of the job workspaces finish removes; None
                removes nothing.
            publishers: The output publishers; `default_publishers()` if omitted.
            io: The EAD io entries, by builder ID.
        """
        self._gateway_factory = gateway_factory
        self._publishers = publishers or default_publishers()
        self._io = io
        self._allowed_api_urls = tuple(url.rstrip("/") for url in allowed_api_urls)
        self._workspace_root = workspace_root

    def identify(self, request: ScheduleRequest) -> SubmissionIdentity:
        """Key admission by installation and job, fingerprinting stable request fields.

        Renewable credentials are left out of the fingerprint.

        Args:
            request: The schedule request.

        Returns:
            The identity keyed `<installation_id>:<job_id>`.

        Raises:
            InvalidScheduleRequestError: If the body is not a valid EMPAIA payload.
        """
        context = schedule_payload(request.body)
        canonical = json.dumps(
            {
                "builder_id": request.builder_id,
                "model_id": request.model_id,
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
    ) -> PreparedInvocation:
        """Admit an EMPAIA job as an invocation of provider inputs.

        Fetches and resolves configuration, records the job's mode, creates provider
        input references, and separates public job state from protected credentials.
        Configuration keys the builder does not take are dropped and logged.
        Optional inputs are taken only if the job has them. Deadlines are capped at
        credential expiry.

        Args:
            request: The schedule request; its body is the EMPAIA schedule payload.
            deadline_at: The requested deadline.

        Returns:
            The invocation with its protected job context; standalone jobs get high
            priority, as someone waits for them.

        Raises:
            InvalidScheduleRequestError: If the payload is invalid, the API URL is not
                allowed, the credential has no valid expiry, the configuration is not
                JSON or not valid for the builder, the invocation fails validation, or
                the App Service rejects a request.
            SchedulingUnavailableError: If the gateway cannot be created or closed, or
                the App Service is temporarily unavailable.
        """
        context = schedule_payload(request.body)
        if not allowed(context.api_url, self._allowed_api_urls):
            raise InvalidScheduleRequestError(
                "EMPAIA api_url is not an allowed AppService"
            )
        credential_expires_at = credential_expiry(context.credential)
        async with preparation_gateway(
            self._gateway_factory,
            context,
        ) as gateway:
            try:
                configuration = await gateway.get_configuration()
            except EmpaiaGatewayError as error:
                raise_preparation_error(error)
            if not is_json_value(configuration):
                raise InvalidScheduleRequestError(
                    "EMPAIA configuration must contain only JSON-compatible values"
                )
            try:
                mode = await gateway.get_mode()
            except EmpaiaGatewayError as error:
                raise_preparation_error(error)
            spec = get_builder(request.builder_id)
            configuration = builder_configuration(spec, configuration, context.job_id)
            names = await job_inputs(spec, gateway)
        effective_deadline = min(deadline_at, credential_expires_at)
        invocation = Invocation(
            inputs={name: ProviderInput(name=name) for name in names},
            resolved_configuration=deepcopy(configuration),
            input_access=InputAccessContext(
                kind=INPUT_ACCESS_KIND,
                state=EmpaiaInputState(
                    api_url=context.api_url, job_id=context.job_id, mode=mode
                ).model_dump(mode="json"),
            ),
            deadline_at=effective_deadline,
        )
        try:
            spec.check_invocation(
                invocation,
                now=effective_deadline - timedelta(microseconds=1),
            )
        except (TypeError, ValueError) as error:
            raise InvalidScheduleRequestError(str(error)) from error
        return PreparedInvocation(
            invocation=invocation,
            protected_context=context.protected_values(),
            deadline_cap=credential_expires_at,
            terminal_report_deadline_at=credential_expires_at,
            # A standalone job was started by hand and someone waits for it;
            # preprocessing runs unattended.
            priority=(
                Priority.HIGH if mode is EmpaiaJobMode.STANDALONE else Priority.NORMAL
            ),
        )

    async def finish(
        self,
        protected_context: dict[str, JsonValue],
        outcome: ProviderFinishOutcome,
        *,
        before_mutation: BeforeMutation | None = None,
    ) -> None:
        """Publish stored outputs and finalize, or report failure/cancellation.

        Every output is prepared before the first write. The workspace is removed
        before a failure or cancellation report, but only after a successful
        publication, since a failed one needs the outputs to retry.

        Args:
            protected_context: The job's stored protected context.
            outcome: The execution's outcome.
            before_mutation: Installed on the gateway to run before each write.

        Raises:
            InvalidScheduleRequestError: If the protected context is malformed.
            ProviderFinishFailureError: If the gateway cannot be created, an output is
                undeclared or cannot be published, or a write fails; its outcome is
                unknown once any output was written or when the gateway says so.
        """
        context = job_context(protected_context)
        if not isinstance(outcome, ProviderSuccess):
            # A failure or cancellation report needs nothing from the workspace.
            await self._remove_workspace(context)
        async with finish_gateway(
            self._gateway_factory,
            context,
        ) as gateway:
            if before_mutation is not None:
                gateway.set_before_mutation(before_mutation)
            mutation_completed = False
            try:
                if isinstance(outcome, ProviderSuccess):
                    builder = get_builder(outcome.builder_id)
                    try:
                        outputs = builder.output_model.model_validate(
                            outcome.result.outputs
                        )
                    except ValidationError as error:
                        raise EmpaiaGatewayError.invalid(
                            "EMPAIA job result does not match its builder's outputs"
                        ) from error
                    declared = builder.outputs
                    io = self._io.get(outcome.builder_id)
                    if io is None:
                        raise EmpaiaGatewayError.invalid(
                            f"builder {outcome.builder_id!r} is not offered in EMPAIA"
                        )
                    # Every output is checked before the first write.
                    commits = []
                    for name, output in dict(outputs).items():
                        spec = declared[name]
                        publisher = self._publishers.publisher_for(
                            name, io.output_type(name), spec, output
                        )
                        try:
                            commit = await publisher.prepare(gateway, name, output)
                        except EmpaiaGatewayError:
                            raise
                        except Exception as error:
                            # Nothing is written yet, and a retry reads the same file.
                            logger.exception("EMPAIA output %s is unreadable", name)
                            raise EmpaiaGatewayError.invalid(
                                f"EMPAIA output {name!r} cannot be published"
                            ) from error
                        commits.append(commit)
                    for commit in commits:
                        await commit()
                        mutation_completed = True
                    await gateway.finalize()
                elif isinstance(outcome, ProviderFailure):
                    await gateway.report_failure(outcome.error.message)
                elif isinstance(outcome, ProviderCancellation):
                    await gateway.report_failure("execution was cancelled")
            except EmpaiaGatewayError as error:
                raise ProviderFinishFailureError(
                    code=FINISH_FAILED,
                    safe_message=error.safe_message,
                    retryable=error.retryable,
                    outcome_unknown=error.outcome_unknown or mutation_completed,
                ) from error
        if isinstance(outcome, ProviderSuccess):
            # Only after publication: a failed one still needs the outputs to retry.
            await self._remove_workspace(context)

    async def _remove_workspace(self, context: EmpaiaJobContext) -> None:
        """Remove every attempt's inputs and outputs of the job.

        Args:
            context: The job whose workspace to remove.
        """
        if self._workspace_root is None:
            return
        directory = job_workspace(self._workspace_root, context.api_url, context.job_id)
        await asyncio.to_thread(shutil.rmtree, directory, True)
