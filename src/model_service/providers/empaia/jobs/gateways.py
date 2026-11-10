"""Opening and closing a job's App Service gateway for admission and for finish."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import TYPE_CHECKING

from model_service.providers.empaia.app_service.gateway import (
    EmpaiaGateway,
    EmpaiaGatewayError,
    EmpaiaGatewayFactory,
    EmpaiaJobContext,
)
from model_service.providers.empaia.jobs.mapping import (
    raise_preparation_error,
)
from model_service.scheduler.public import (
    ProviderFinishFailureError,
    SchedulingUnavailableError,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class EmpaiaErrorCode(StrEnum):
    """Name the safe failure for unsuccessful EMPAIA publication or terminal reporting."""

    FINISH_FAILED = "empaia_finish_failed"


logger = logging.getLogger(__name__)


@asynccontextmanager
async def preparation_gateway(
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
        raise_preparation_error(error)
    except Exception as error:
        raise SchedulingUnavailableError(
            "EMPAIA integration is temporarily unavailable"
        ) from error
    try:
        yield gateway
    except BaseException:
        await close_preserving_outcome(gateway)
        raise
    try:
        await gateway.close()
    except EmpaiaGatewayError as error:
        raise_preparation_error(error)
    except Exception as error:
        raise SchedulingUnavailableError(
            "EMPAIA integration is temporarily unavailable"
        ) from error


@asynccontextmanager
async def finish_gateway(
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
        raise finish_failure(error) from error
    except Exception as error:
        raise ProviderFinishFailureError(
            code=EmpaiaErrorCode.FINISH_FAILED,
            safe_message="EMPAIA integration is temporarily unavailable",
            retryable=True,
        ) from error
    try:
        yield gateway
    except BaseException:
        await close_preserving_outcome(gateway)
        raise
    try:
        await gateway.close()
    except Exception as error:
        logger.warning("EMPAIA gateway cleanup failed: %s", type(error).__name__)


async def close_preserving_outcome(gateway: EmpaiaGateway) -> None:
    """Close a gateway and log cleanup errors without replacing an already established outcome.

    Args:
        gateway: The gateway to close.
    """
    try:
        await gateway.close()
    except Exception as error:
        logger.warning("EMPAIA gateway cleanup failed: %s", type(error).__name__)


def finish_failure(error: EmpaiaGatewayError) -> ProviderFinishFailureError:
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
