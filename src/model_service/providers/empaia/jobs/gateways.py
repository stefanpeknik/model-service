"""Opening and closing a job's App Service gateway for admission and for finish."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Never

from model_service.providers.empaia.app_service.gateway import EmpaiaGatewayError
from model_service.providers.empaia.jobs.mapping import raise_preparation_error
from model_service.scheduler.public import (
    ProviderFinishFailureError,
    SchedulingUnavailableError,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from contextlib import AbstractAsyncContextManager

    from model_service.providers.empaia.app_service.gateway import (
        EmpaiaGateway,
        EmpaiaGatewayFactory,
        EmpaiaJobContext,
    )


# The safe error code for unsuccessful EMPAIA publication or terminal reporting.
FINISH_FAILED = "empaia_finish_failed"
_UNAVAILABLE = "EMPAIA integration is temporarily unavailable"

logger = logging.getLogger(__name__)


def preparation_gateway(
    factory: EmpaiaGatewayFactory, context: EmpaiaJobContext
) -> AbstractAsyncContextManager[EmpaiaGateway]:
    """Own a gateway for admission; failing to open or close it refuses admission.

    Args:
        factory: Creates the gateway.
        context: The job's connection settings.

    Returns:
        A context manager yielding the gateway.
    """
    return _owned_gateway(factory, context, _raise_preparation, close_strictly=True)


def finish_gateway(
    factory: EmpaiaGatewayFactory, context: EmpaiaJobContext
) -> AbstractAsyncContextManager[EmpaiaGateway]:
    """Own a gateway for finish; closing errors are only logged, keeping the outcome.

    Args:
        factory: Creates the gateway.
        context: The job's connection settings.

    Returns:
        A context manager yielding the gateway.
    """
    return _owned_gateway(factory, context, _raise_finish, close_strictly=False)


@asynccontextmanager
async def _owned_gateway(
    factory: EmpaiaGatewayFactory,
    context: EmpaiaJobContext,
    raise_translated: Callable[[Exception], Never],
    *,
    close_strictly: bool,
) -> AsyncIterator[EmpaiaGateway]:
    """Open, yield and close a gateway, translating opening failures.

    A failure in the body closes the gateway, logging cleanup errors, and is
    re-raised unchanged.

    Args:
        factory: Creates the gateway.
        context: The job's connection settings.
        raise_translated: Raises the caller-facing error for a gateway failure.
        close_strictly: Whether a failed close after success is translated too.

    Yields:
        The gateway.
    """
    try:
        gateway = factory(context)
    except Exception as error:
        raise_translated(error)
    try:
        yield gateway
    except BaseException:
        await _close_logged(gateway)
        raise
    if not close_strictly:
        await _close_logged(gateway)
        return
    try:
        await gateway.close()
    except Exception as error:
        raise_translated(error)


async def _close_logged(gateway: EmpaiaGateway) -> None:
    """Close a gateway, logging rather than raising a cleanup error.

    Args:
        gateway: The gateway to close.
    """
    try:
        await gateway.close()
    except Exception as error:
        logger.warning("EMPAIA gateway cleanup failed: %s", type(error).__name__)


def _raise_preparation(error: Exception) -> Never:
    """Refuse admission: gateway errors by their retry flag, anything else as unavailable.

    Args:
        error: The failure.

    Raises:
        SchedulingUnavailableError: If the failure is retryable or unexpected.
        InvalidScheduleRequestError: If the gateway error is not retryable.
    """
    if isinstance(error, EmpaiaGatewayError):
        raise_preparation_error(error)
    raise SchedulingUnavailableError(_UNAVAILABLE) from error


def _raise_finish(error: Exception) -> Never:
    """Fail the finish with the gateway's safe metadata, or as retryable if unexpected.

    Args:
        error: The failure.

    Raises:
        ProviderFinishFailureError: Always.
    """
    if not isinstance(error, EmpaiaGatewayError):
        raise ProviderFinishFailureError(
            code=FINISH_FAILED, safe_message=_UNAVAILABLE, retryable=True
        ) from error
    raise ProviderFinishFailureError(
        code=FINISH_FAILED,
        safe_message=error.safe_message,
        retryable=error.retryable,
        outcome_unknown=error.outcome_unknown,
    ) from error
