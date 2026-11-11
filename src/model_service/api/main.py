"""The production FastAPI application Uvicorn imports."""

import logging

from model_service.api.app_factory import build_api_process


# Uvicorn configures only its own loggers before importing this module; without
# its handler, model-service logs below WARNING never reach the pod log.
_package_logger = logging.getLogger("model_service")
_package_logger.handlers = logging.getLogger("uvicorn").handlers
_package_logger.setLevel(logging.INFO)
_package_logger.propagate = False

app = build_api_process()
