"""What every model-service log line says about where it comes from.

Lines are JSON objects. Entry points set the execution a coroutine works for
in `EXECUTION_ID`: the control plane's scheduler where it admits, claims and finishes an
execution, the pipeline runner once per run. `ExecutionIdFilter` adds it to each
record, so no function passes ids around. The control plane also adds the HTTP request's
id, as `request_id`.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator

    from pythonjsonlogger.json import JsonFormatter


# The execution the current coroutine works for; asyncio tasks inherit it.
EXECUTION_ID: ContextVar[str | None] = ContextVar("execution_id", default=None)


@contextmanager
def logging_execution(execution_id: str | None) -> Iterator[None]:
    """Name an execution in the log lines of the block and the tasks it starts.

    Args:
        execution_id: The execution; None leaves the lines without one.

    Yields:
        Nothing; the previous execution is restored on exit.
    """
    token = EXECUTION_ID.set(execution_id)
    try:
        yield
    finally:
        EXECUTION_ID.reset(token)


class ExecutionIdFilter(logging.Filter):
    """Add the current execution's id to each record, as `execution_id`."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Set the record's `execution_id`; null outside an execution.

        Args:
            record: The record to label.

        Returns:
            True: every record is kept.
        """
        record.execution_id = EXECUTION_ID.get()
        return True


def json_formatter() -> JsonFormatter:
    """Build the JSON formatter every model-service process logs with.

    Attributes filters add, such as `execution_id`, become fields too.

    Returns:
        A formatter writing `time`, `level`, `logger` and `message`.
    """
    from pythonjsonlogger.json import JsonFormatter

    return JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={
            "asctime": "time",
            "levelname": "level",
            "name": "logger",
            "correlation_id": "request_id",
        },
    )
