"""How an EMPAIA job maps onto a builder: its configuration and its inputs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Never

from model_service.api.scheduler.public import (
    InvalidScheduleRequestError,
    SchedulingUnavailableError,
)
from model_service.providers.empaia.app_service.gateway import EmpaiaGatewayError
from model_service.providers.empaia.ead.apps import VIEWER_CONFIG_KEY


if TYPE_CHECKING:
    from model_service.builders.declarations import BuilderSpec
    from model_service.contracts.json_values import JsonValue
    from model_service.providers.empaia.app_service.gateway import EmpaiaGateway


logger = logging.getLogger(__name__)


def known_settings(
    spec: BuilderSpec, configuration: dict[str, JsonValue], job_id: str
) -> dict[str, JsonValue]:
    """Take the settings of a job's app configuration the builder knows.

    Installation admins may configure an app with keys meant for something
    else; the builder takes only the ones it knows, and the rest are logged.
    The viewer's settings are expected and meant for the case browser. The
    values are not checked here: the attempt checks them, so a bad one ends
    the job with a message for the user (EMPAIA's failure endpoint).

    Args:
        spec: The builder the job runs.
        configuration: The app configuration EMPAIA gave the job.
        job_id: The job, for the log.

    Returns:
        The settings the builder takes, as given.
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
    return {k: v for k, v in configuration.items() if k in known}


async def job_inputs(spec: BuilderSpec, gateway: EmpaiaGateway) -> list[str]:
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
            raise_preparation_error(error)
    return names


def raise_preparation_error(error: EmpaiaGatewayError) -> Never:
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
