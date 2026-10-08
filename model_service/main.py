"""Expose the production FastAPI application imported by Uvicorn.

Importing this module reads required environment settings and builds adapters. Database
initialization and worker startup occur in the application lifespan.
"""

from model_service.runtime import create_app


app = create_app()
