"""Log a builder replica's model-service lines as JSON on stdout.

Serve configures only its own logger, and in Ray 2.58 its stream handler
writes text whatever the encoding, which applies to its files only. So the
`model_service` logger gets its own JSON stdout handler, which names the run's
execution, and keeps Serve's file handler for the Ray dashboard.
"""

from __future__ import annotations

import logging
import sys

from model_service.contracts.log_context import ExecutionIdFilter, json_formatter


def configure_replica_logging() -> None:
    """Send `model_service` logs at INFO and above to stdout as JSON and to Serve's files."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(json_formatter())
    handler.addFilter(ExecutionIdFilter())
    serve_files = [
        h
        for h in logging.getLogger("ray.serve").handlers
        if not isinstance(h, logging.StreamHandler)
    ]
    package_logger = logging.getLogger("model_service")
    package_logger.handlers = [handler, *serve_files]
    package_logger.setLevel(logging.INFO)
    package_logger.propagate = False
