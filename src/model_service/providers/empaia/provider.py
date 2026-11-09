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
import math
import shutil
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Never

import jwt
from pydantic import Field, ValidationError

from model_service.builders.contracts import (
    InputAccessContext,
    InvalidInvocation,
    Invocation,
    ProviderInput,
    is_json_value,
)
from model_service.builders.registry import get_builder, parse_result
from model_service.providers.empaia.apps import VIEWER_CONFIG_KEY
from model_service.providers.empaia.gateway import (
    INPUT_ACCESS_KIND,
    PROVIDER_NAME,
    EmpaiaGateway,
    EmpaiaGatewayError,
    EmpaiaGatewayFactory,
    EmpaiaInputState,
    EmpaiaJobContext,
    EmpaiaJobMode,
)
from model_service.providers.empaia.input_access import job_workspace
from model_service.providers.empaia.io import BUILDER_IO
from model_service.providers.empaia.publishers import default_publishers
from model_service.scheduler.contracts import (
    InvalidScheduleRequestError,
    ScheduleRequest,
    SchedulingUnavailableError,
)
from model_service.scheduler.domain import (
    PreparedInvocation,
    Priority,
    SubmissionIdentity,
)
from model_service.scheduler.providers import (
    ProviderAdapter,
    ProviderCancellation,
    ProviderFailure,
    ProviderFinishFailureError,
    ProviderFinishOutcome,
    ProviderSuccess,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from pathlib import Path

    from model_service.builders.contracts import JsonValue
    from model_service.builders.spec import BuilderSpec
    from model_service.providers.empaia.io import BuilderIo
    from model_service.providers.empaia.publishers import PublisherRegistry
    from model_service.scheduler.providers import BeforeMutation


class EmpaiaErrorCode(StrEnum):
    """Name the safe failure for unsuccessful EMPAIA publication or terminal reporting."""

    FINISH_FAILED = "empaia_finish_failed"


class EmpaiaSchedulePayload(EmpaiaJobContext):
    """Require an installation ID alongside the job origin, ID, and private credential."""

    installation_id: Annotated[str, Field(min_length=1)]


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
        context = _schedule_payload(request.body)
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
        Configuration keys the builder does not take are dropped and logged, except
        the viewer's. Optional inputs are taken only if the job has them. Deadlines
        are capped at credential expiry.

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
        context = _schedule_payload(request.body)
        if not _allowed(context.api_url, self._allowed_api_urls):
            raise InvalidScheduleRequestError(
                "EMPAIA api_url is not an allowed AppService"
            )
        credential_expires_at = _credential_expiry(context.credential)
        async with _preparation_gateway(
            self._gateway_factory,
            context,
        ) as gateway:
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
                mode = await gateway.get_mode()
            except EmpaiaGatewayError as error:
                _raise_preparation_error(error)
            spec = get_builder(request.builder_id)
            configuration = _builder_configuration(spec, configuration, context.job_id)
            names = await _job_inputs(spec, gateway)
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
        context = _job_context(protected_context)
        if not isinstance(outcome, ProviderSuccess):
            # A failure or cancellation report needs nothing from the workspace.
            await self._remove_workspace(context)
        async with _finish_gateway(
            self._gateway_factory,
            context,
        ) as gateway:
            if before_mutation is not None:
                gateway.set_before_mutation(before_mutation)
            mutation_completed = False
            try:
                if isinstance(outcome, ProviderSuccess):
                    try:
                        outputs = parse_result(outcome.builder_id, outcome.result)
                    except ValidationError as error:
                        raise EmpaiaGatewayError.invalid(
                            "EMPAIA job result does not match its builder's outputs"
                        ) from error
                    declared = get_builder(outcome.builder_id).outputs
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
                    code=EmpaiaErrorCode.FINISH_FAILED,
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


@asynccontextmanager
async def _preparation_gateway(
    factory: EmpaiaGatewayFactory,
    context: EmpaiaJobContext,
) -> AsyncIterator[EmpaiaGateway]:
    """Own a preparation gateway and translate acquisition/cleanup failures.

    A failure in the body closes the gateway, logging cleanup errors, and is
    re-raised unchanged.

    Args:
        factory: Creates the gateway.
        context: The job's connection settings.

    Yields:
        The gateway.

    Raises:
        SchedulingUnavailableError: If creating or closing the gateway fails
            retryably or unexpectedly.
        InvalidScheduleRequestError: If creating or closing the gateway fails
            with a non-retryable gateway error.
    """
    try:
        gateway = factory(context)
    except EmpaiaGatewayError as error:
        _raise_preparation_error(error)
    except Exception as error:
        raise SchedulingUnavailableError(
            "EMPAIA integration is temporarily unavailable"
        ) from error
    try:
        yield gateway
    except BaseException:
        await _close_preserving_outcome(gateway)
        raise
    try:
        await gateway.close()
    except EmpaiaGatewayError as error:
        _raise_preparation_error(error)
    except Exception as error:
        raise SchedulingUnavailableError(
            "EMPAIA integration is temporarily unavailable"
        ) from error


@asynccontextmanager
async def _finish_gateway(
    factory: EmpaiaGatewayFactory,
    context: EmpaiaJobContext,
) -> AsyncIterator[EmpaiaGateway]:
    """Own a finish gateway and translate acquisition failures.

    A failure in the body closes the gateway and is re-raised unchanged; cleanup
    errors are only logged, so the external completion outcome is kept.

    Args:
        factory: Creates the gateway.
        context: The job's connection settings.

    Yields:
        The gateway.

    Raises:
        ProviderFinishFailureError: If creating the gateway fails; retryable when
            the failure is not a gateway error.
    """
    try:
        gateway = factory(context)
    except EmpaiaGatewayError as error:
        raise _finish_failure(error) from error
    except Exception as error:
        raise ProviderFinishFailureError(
            code=EmpaiaErrorCode.FINISH_FAILED,
            safe_message="EMPAIA integration is temporarily unavailable",
            retryable=True,
        ) from error
    try:
        yield gateway
    except BaseException:
        await _close_preserving_outcome(gateway)
        raise
    try:
        await gateway.close()
    except Exception as error:
        logger.warning("EMPAIA gateway cleanup failed: %s", type(error).__name__)


async def _close_preserving_outcome(gateway: EmpaiaGateway) -> None:
    """Close a gateway and log cleanup errors without replacing an already established outcome.

    Args:
        gateway: The gateway to close.
    """
    try:
        await gateway.close()
    except Exception as error:
        logger.warning("EMPAIA gateway cleanup failed: %s", type(error).__name__)


def _raise_preparation_error(error: EmpaiaGatewayError) -> Never:
    """Map a gateway error to the scheduler's admission errors.

    Args:
        error: The gateway error.

    Raises:
        SchedulingUnavailableError: If the error is retryable.
        InvalidScheduleRequestError: Otherwise.
    """
    if error.retryable:
        raise SchedulingUnavailableError(error.safe_message) from error
    raise InvalidScheduleRequestError(error.safe_message) from error


def _finish_failure(error: EmpaiaGatewayError) -> ProviderFinishFailureError:
    """Translate safe gateway retry and uncertainty metadata into a provider finish failure.

    Args:
        error: The gateway error.

    Returns:
        The finish failure, with the same message, retry flag and outcome.
    """
    return ProviderFinishFailureError(
        code=EmpaiaErrorCode.FINISH_FAILED,
        safe_message=error.safe_message,
        retryable=error.retryable,
        outcome_unknown=error.outcome_unknown,
    )


def _allowed(api_url: str, allowed: tuple[str, ...]) -> bool:
    """Check whether an API URL equals or lies below an allowed prefix.

    Trailing slashes are ignored, and a prefix matches whole path segments only.

    Args:
        api_url: The job's App Service URL.
        allowed: The allowed prefixes, without trailing slashes.

    Returns:
        True if the URL is allowed.
    """
    url = api_url.rstrip("/")
    return any(url == prefix or url.startswith(prefix + "/") for prefix in allowed)


def _schedule_payload(values: dict[str, JsonValue]) -> EmpaiaSchedulePayload:
    """Validate provider admission fields, hiding credential-bearing validation input from public errors.

    Args:
        values: The schedule request body.

    Returns:
        The schedule payload.

    Raises:
        InvalidScheduleRequestError: If the body is not a valid payload.
    """
    try:
        return EmpaiaSchedulePayload.model_validate(values)
    except ValidationError as error:
        raise InvalidScheduleRequestError(
            "EMPAIA schedule payload is invalid"
        ) from error


def _job_context(values: dict[str, JsonValue]) -> EmpaiaJobContext:
    """Validate a stored protected job context.

    Args:
        values: The stored protected context.

    Returns:
        The job context.

    Raises:
        InvalidScheduleRequestError: If the context is malformed; the message
            holds no context values.
    """
    try:
        return EmpaiaJobContext.model_validate(values)
    except ValidationError as error:
        raise InvalidScheduleRequestError(
            "EMPAIA protected job context is invalid"
        ) from error


def _credential_expiry(credential: str) -> datetime:
    """Read the numeric `exp` claim of the credential to cap deadlines.

    This parses the JWT payload without verifying its signature.

    Args:
        credential: The job's token.

    Returns:
        The expiry, in UTC.

    Raises:
        InvalidScheduleRequestError: If the credential is not a JWT or its `exp`
            is missing, not a finite number, or out of range.
    """
    try:
        claims = jwt.decode(
            credential,
            options={
                "verify_signature": False,
                "verify_exp": False,
                "require": ["exp"],
            },
        )
        expires_at = claims["exp"]
        if (
            not isinstance(expires_at, int | float)
            or isinstance(expires_at, bool)
            or not math.isfinite(expires_at)
        ):
            raise TypeError
        return datetime.fromtimestamp(expires_at, tz=UTC)
    except (
        jwt.InvalidTokenError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        OSError,
    ) as error:
        raise InvalidScheduleRequestError(
            "EMPAIA credential does not contain a valid expiry"
        ) from error


def _builder_configuration(
    spec: BuilderSpec, configuration: dict[str, JsonValue], job_id: str
) -> dict[str, JsonValue]:
    """Resolve a job's app configuration into the builder's configuration.

    Installation admins may configure an app with keys meant for something
    else; the builder takes only the ones it knows, and the rest are logged.
    The viewer's settings are expected and meant for the case browser.

    Args:
        spec: The builder the job runs.
        configuration: The app configuration EMPAIA gave the job.
        job_id: The job, for the log.

    Returns:
        The builder's resolved configuration, as JSON.

    Raises:
        InvalidScheduleRequestError: If the builder rejects a value.
    """
    known = spec.configuration.model_fields
    if ignored := sorted(configuration.keys() - known.keys() - {VIEWER_CONFIG_KEY}):
        logger.warning(
            "EMPAIA job %s: ignoring configuration keys the %s builder "
            "does not take: %s",
            job_id,
            spec.id,
            ", ".join(ignored),
        )
    taken = {k: v for k, v in configuration.items() if k in known}
    try:
        return spec.resolve_configuration(taken).model_dump(mode="json")
    except InvalidInvocation as error:
        raise InvalidScheduleRequestError(str(error)) from error


async def _job_inputs(spec: BuilderSpec, gateway: EmpaiaGateway) -> list[str]:
    """Name the builder inputs the job supplies.

    Required inputs always; optional ones only if the job has them, so a
    preprocessing app, for example, gets only a slide.

    Args:
        spec: The builder the job runs.
        gateway: The job's gateway.

    Returns:
        The input names, in declaration order.

    Raises:
        InvalidScheduleRequestError: If the App Service rejects the lookup.
        SchedulingUnavailableError: If it is temporarily unavailable.
    """
    names = []
    for name, input_spec in spec.inputs.items():
        try:
            if input_spec.required or await gateway.has_input(name):
                names.append(name)
        except EmpaiaGatewayError as error:
            _raise_preparation_error(error)
    return names
