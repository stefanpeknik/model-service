"""Translate EMPAIA jobs into durable executions and publish their stored outcomes.

A job runs a pipeline (`/v1/empaia/pipelines/{id}`). Admission reads the job's
configuration and parameters, records each input's EAD type and how each
output is published, and caps deadlines at credential expiry. Finish
publishes through the recorded publishers: it validates the result,
prepares every output, commits them in order, and then finalizes. Finish
retries are separate from computation and stop when publication's external
outcome is uncertain.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError

from components import EmpaiaInput, EmpaiaOutput
from model_service.contracts.invocation import (
    InputAccessContext,
    Invocation,
    ProviderInput,
)
from model_service.contracts.json_values import is_json_value
from model_service.contracts.result import SharedArtifact
from model_service.control_plane.scheduler.public import (
    InvalidScheduleRequestError,
    PipelineScheduleRequest,
    PreparedInvocation,
    Priority,
    ProviderCancellation,
    ProviderFailure,
    ProviderFinishFailureError,
    ProviderFinishOutcome,
    ProviderSuccess,
    SubmissionIdentity,
)
from model_service.directories import remove_directory
from model_service.formats.known import format_of
from model_service.providers.empaia.app_service.gateway import (
    INPUT_ACCESS_KIND,
    PROVIDER_NAME,
    AdmittedEmpaiaJobContext,
    EmpaiaGatewayError,
    EmpaiaGatewayFactory,
    EmpaiaInputState,
    EmpaiaJobContext,
    EmpaiaJobMode,
    Publications,
)
from model_service.providers.empaia.jobs.failure_message import fit_failure_message
from model_service.providers.empaia.jobs.gateways import (
    FINISH_FAILED,
    finish_gateway,
    preparation_gateway,
)
from model_service.providers.empaia.jobs.inputs import job_workspace
from model_service.providers.empaia.jobs.mapping import (
    job_inputs,
    known_settings,
    raise_preparation_error,
)
from model_service.providers.empaia.jobs.payload import (
    allowed,
    credential_expiry,
    job_context,
    schedule_payload,
)
from model_service.providers.provider import ProviderAdapter
from model_service.registry.empaia import converter, publisher


if TYPE_CHECKING:
    from collections.abc import Callable

    from components import Commit
    from model_service.catalog import Catalog
    from model_service.catalog.pipelines import Pipeline
    from model_service.contracts.json_values import JsonValue
    from model_service.contracts.result import RunResult
    from model_service.control_plane.scheduler.public import BeforeMutation
    from model_service.providers.empaia.app_service.gateway import (
        ComponentUse,
        EmpaiaGateway,
    )


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Admitted:
    """What the job runs with: settings, inputs and their EAD types, and publications."""

    configuration: dict[str, JsonValue]
    inputs: dict[str, ComponentUse]
    publications: Publications


class EmpaiaProviderAdapter(ProviderAdapter):
    """Implement provider-scoped admission and terminal publication for EMPAIA jobs."""

    name = PROVIDER_NAME

    def __init__(
        self,
        gateway_factory: EmpaiaGatewayFactory,
        *,
        allowed_api_urls: tuple[str, ...],
        workspace_root: Path | None = None,
        catalog: Callable[[], Catalog] | None = None,
    ) -> None:
        """Only `allowed_api_urls` origins or path prefixes receive job credentials; none, no job.

        Args:
            gateway_factory: Creates the gateway of each job.
            allowed_api_urls: App Service origins or path prefixes allowed to receive
                job credentials.
            workspace_root: The root of the job workspaces finish removes; None
                removes nothing.
            catalog: The deployed catalog, read at each admission; None
                offers no pipeline.
        """
        self._gateway_factory = gateway_factory
        self._catalog = catalog
        self._allowed_api_urls = tuple(url.rstrip("/") for url in allowed_api_urls)
        self._workspace_root = workspace_root

    def identify(self, request: PipelineScheduleRequest) -> SubmissionIdentity:
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
                "pipeline_id": request.pipeline_id,
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
        request: PipelineScheduleRequest,
        deadline_at: datetime,
    ) -> PreparedInvocation:
        """Admit an EMPAIA job as an invocation of provider inputs.

        Fetches the configuration, and for a standalone job the values of
        its app's parameters, which win over it; records the
        job's mode and each input's EAD type, and separates public job state
        from protected credentials and publications. Configuration keys
        that are no settings are dropped and logged, except the viewer's.
        Optional inputs are taken only if the job has them. Deadlines are
        capped at credential expiry.

        Args:
            request: The schedule request; its body is the EMPAIA schedule payload.
            deadline_at: The requested deadline.

        Returns:
            The invocation with its protected job context; standalone jobs get high
            priority, as someone waits for them.

        Raises:
            InvalidScheduleRequestError: If the payload is invalid, the API URL is not
                allowed, the credential has no valid expiry, the pipeline is not
                offered in EMPAIA, the configuration is not JSON, the invocation
                fails validation, or the App Service rejects a request.
            SchedulingUnavailableError: If the gateway cannot be created or closed, or
                the App Service is temporarily unavailable.
        """
        context = schedule_payload(request.body)
        if not allowed(context.api_url, self._allowed_api_urls):
            raise InvalidScheduleRequestError(
                "EMPAIA api_url is not an allowed AppService"
            )
        credential_expires_at = credential_expiry(context.credential)
        pipeline = self._pipeline(request.pipeline_id)
        async with preparation_gateway(self._gateway_factory, context) as gateway:
            try:
                configuration = await gateway.get_configuration()
                if not is_json_value(configuration):
                    raise InvalidScheduleRequestError(
                        "EMPAIA configuration must contain only JSON-compatible values"
                    )
                mode = await gateway.get_mode()
                admitted = await _admit_pipeline(
                    pipeline, configuration, mode, gateway, context.job_id
                )
            except EmpaiaGatewayError as error:
                raise_preparation_error(error)
        effective_deadline = min(deadline_at, credential_expires_at)
        invocation = Invocation(
            inputs={name: ProviderInput(name=name) for name in admitted.inputs},
            configuration=deepcopy(admitted.configuration),
            input_access=InputAccessContext(
                kind=INPUT_ACCESS_KIND,
                state=EmpaiaInputState(
                    api_url=context.api_url,
                    job_id=context.job_id,
                    mode=mode,
                    inputs=admitted.inputs,
                ).model_dump(mode="json"),
            ),
            deadline_at=effective_deadline,
        )
        protected = AdmittedEmpaiaJobContext(
            api_url=context.api_url,
            job_id=context.job_id,
            credential=context.credential,
            publications=admitted.publications,
        )
        return PreparedInvocation(
            invocation=invocation,
            protected_context=protected.protected_values(),
            deadline_cap=credential_expires_at,
            terminal_report_deadline_at=credential_expires_at,
            # A standalone job was started by hand and someone waits for it;
            # preprocessing runs unattended.
            priority=(
                Priority.HIGH if mode is EmpaiaJobMode.STANDALONE else Priority.NORMAL
            ),
        )

    def _pipeline(self, pipeline_id: str) -> Pipeline:
        """The deployed pipeline a job runs, which must be offered in EMPAIA.

        Args:
            pipeline_id: The pipeline the job's app runs.

        Returns:
            The pipeline.

        Raises:
            InvalidScheduleRequestError: It is not deployed or has no `providers.empaia`.
        """
        found = self._catalog().pipelines if self._catalog else {}
        pipeline = found.get(pipeline_id)
        if pipeline is None or pipeline.empaia is None:
            raise InvalidScheduleRequestError(
                f"pipeline {pipeline_id!r} is not offered in EMPAIA"
            )
        return pipeline

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
        async with finish_gateway(self._gateway_factory, context) as gateway:
            if before_mutation is not None:
                gateway.set_before_mutation(before_mutation)
            mutation_completed = False
            try:
                if isinstance(outcome, ProviderSuccess):
                    publications = self._publications(context, outcome)
                    # Every output is checked before the first write.
                    commits = [
                        await _prepare(gateway, name, use, publications.slide, output)
                        for name, use, output in _published(
                            publications, outcome.result
                        )
                    ]
                    for commit in commits:
                        await commit()
                        mutation_completed = True
                    await gateway.finalize()
                elif isinstance(outcome, ProviderFailure):
                    await gateway.report_failure(
                        fit_failure_message(outcome.error.message)
                    )
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

    def _publications(
        self, context: AdmittedEmpaiaJobContext, outcome: ProviderSuccess
    ) -> Publications:
        """How a result is published, as admitted.

        The result must have the pipeline's outputs, no more.

        Args:
            context: The job's context, with the publications admission recorded.
            outcome: The successful processing outcome.

        Returns:
            The recorded publications.

        Raises:
            EmpaiaGatewayError: Not retryable if the result doesn't match the
                pipeline.
        """
        sinks = {
            sink["output"]
            for sink in cast("list[dict[str, Any]]", outcome.target.pipeline["sinks"])
        }
        if set(outcome.result.outputs) != sinks:
            raise EmpaiaGatewayError.invalid(
                "EMPAIA job result does not match its pipeline's outputs"
            )
        return context.publications

    async def _remove_workspace(self, context: EmpaiaJobContext) -> None:
        """Remove every attempt's inputs and outputs of the job.

        Args:
            context: The job whose workspace to remove.
        """
        if self._workspace_root is None:
            return
        directory = job_workspace(self._workspace_root, context.api_url, context.job_id)
        await asyncio.to_thread(remove_directory, directory)


async def _admit_pipeline(
    pipeline: Pipeline,
    configuration: dict[str, JsonValue],
    mode: EmpaiaJobMode,
    gateway: EmpaiaGateway,
    job_id: str,
) -> _Admitted:
    """Admit a job of a pipeline's app: its settings, parameters, inputs and publications.

    The scheduler resolves the pipeline with the settings, so a bad value
    refuses the job.

    Args:
        pipeline: The pipeline the app runs.
        configuration: The job's provider configuration.
        mode: How the job was started.
        gateway: The job's gateway, for parameters and input presence.
        job_id: The EMPAIA job.

    Returns:
        The settings, inputs with their EAD types, and publications.

    Raises:
        EmpaiaGatewayError: A parameter or input lookup failed.
    """
    app = pipeline.empaia
    assert app is not None
    settings = known_settings(
        pipeline.settings(), f"pipeline {pipeline.id}", configuration, job_id
    )
    if mode is EmpaiaJobMode.STANDALONE:
        schema = pipeline.settings_schema()["properties"]
        primitive = converter({"type": "primitive"})
        for name in app.parameters:
            item = EmpaiaInput(name, app.slide, schema[name])
            settings[name] = await primitive.materialize(gateway, item, Path())  # type: ignore[assignment]
    names = await job_inputs(
        {name: pipeline.inputs[name] for name in app.inputs}, gateway
    )
    return _Admitted(
        configuration=settings,
        inputs={name: _use(app.inputs[name]) for name in names},
        publications=Publications(
            slide=app.slide,
            outputs={
                name: p.options.model_dump(mode="json")
                for name, p in app.outputs.items()
            },
        ),
    )


def _use(component: Any) -> ComponentUse:
    """A converter as admission records it: its EAD type and options.

    Args:
        component: A converter or publisher.

    Returns:
        `type` and every option.
    """
    return {"type": component.ead_type, **component.options.model_dump(mode="json")}


def _published(
    publications: Publications, result: RunResult
) -> list[tuple[str, ComponentUse, Any]]:
    """Each output to publish with its recorded publisher and stored output.

    The result was checked to have every one.

    Args:
        publications: The publications admission recorded.
        result: The run's result.

    Returns:
        `(name, publisher use, stored output)` per published output.
    """
    return [
        (name, use, result.outputs[name]) for name, use in publications.outputs.items()
    ]


async def _prepare(
    gateway: EmpaiaGateway,
    name: str,
    use: ComponentUse,
    slide: str,
    output: Any,
) -> Commit:
    """Pick an output's publisher by its stored format and prepare it.

    Args:
        gateway: The job's gateway.
        name: The output's EAD io key.
        use: The publisher's recorded EAD type and options.
        slide: The app's slide input key.
        output: The stored output.

    Returns:
        The commit that publishes it.

    Raises:
        EmpaiaGatewayError: Not retryable if the output is missing, not a file
            of the format the publisher reads, or unreadable.
    """
    found = (
        format_of(output.media_type)
        if isinstance(output, SharedArtifact) and output.media_type
        else None
    )
    if found is None:
        raise EmpaiaGatewayError.invalid(
            f"EMPAIA output {name!r} is not a stored file of a known format"
        )
    try:
        chosen = publisher(use, found.name)
    except (LookupError, ValidationError) as error:
        raise EmpaiaGatewayError.invalid(
            f"EMPAIA output {name!r} has no publisher"
        ) from error
    try:
        return await chosen.prepare(
            gateway, EmpaiaOutput(name, slide), Path(output.path), output.media_type
        )
    except EmpaiaGatewayError:
        raise
    except Exception as error:
        # Nothing is written yet, and a retry reads the same file.
        logger.exception("EMPAIA output %s is unreadable", name)
        raise EmpaiaGatewayError.invalid(
            f"EMPAIA output {name!r} cannot be published"
        ) from error
