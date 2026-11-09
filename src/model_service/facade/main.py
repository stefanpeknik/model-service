"""Expose the production FastAPI application imported by Uvicorn.

Importing this module reads required environment settings and builds adapters. Database
initialization and worker startup occur in the application lifespan.
"""

import logging

from model_service.facade.app import create_app


# Uvicorn configures only its own loggers before importing this module; without
# its handler, model-service logs below WARNING never reach the pod log.
_package_logger = logging.getLogger("model_service")
_package_logger.handlers = logging.getLogger("uvicorn").handlers
_package_logger.setLevel(logging.INFO)
_package_logger.propagate = False

app = create_app()
