"""How an EMPAIA job maps onto what it runs: its configuration and its inputs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Never

from model_service.api.scheduler.public import (
    InvalidScheduleRequestError,
    SchedulingUnavailableError,
)
from model_service.providers.empaia.app_service.gateway import EmpaiaGatewayError
from model_service.providers.empaia.ead.vocabulary import VIEWER_CONFIG_KEY


if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from model_service.contracts.json_values import JsonValue
    from model_service.providers.empaia.app_service.gateway import EmpaiaGateway


logger = logging.getLogger(__name__)


def known_settings(
    known: Collection[str],
    runs: str,
    configuration: dict[str, JsonValue],
    job_id: str,
) -> dict[str, JsonValue]:
    """Take the settings of a job's app configuration that what it runs knows.

    Installation admins may configure an app with keys meant for something
    else; only known ones are taken, and the rest are logged. The viewer's
    settings are expected and meant for the case browser.

    Args:
        known: The setting names of the analysis or builder.
        runs: What the job runs, for the log, e.g. `analysis prostate-heatmap`.
        configuration: The app configuration EMPAIA gave the job.
        job_id: The job, for the log.

    Returns:
        The known settings, as given.
    """
    if ignored := sorted(configuration.keys() - set(known) - {VIEWER_CONFIG_KEY}):
        logger.warning(
            "EMPAIA job %s: ignoring configuration keys the %s does not take: %s",
            job_id,
            runs,
            ", ".join(ignored),
        )
    return {k: v for k, v in configuration.items() if k in known}


async def job_inputs(inputs: Mapping[str, bool], gateway: EmpaiaGateway) -> list[str]:
    """Name the inputs the job supplies.

    Required inputs always; optional ones only if the job has them, so a
    preprocessing job, for example, gets only a slide.

    Args:
        inputs: Each input and whether it is required, in declaration order.
        gateway: The job's gateway.

    Returns:
        The input names, in declaration order.

    Raises:
        InvalidScheduleRequestError: If the App Service rejects the lookup.
        SchedulingUnavailableError: If it is temporarily unavailable.
    """
    names = []
    for name, required in inputs.items():
        try:
            if required or await gateway.has_input(name):
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
