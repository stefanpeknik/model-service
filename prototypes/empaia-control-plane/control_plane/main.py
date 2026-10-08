"""Expose the historical SQLite prototype FastAPI application.

Import reads prototype environment settings and constructs its store and service. The
current production HTTP entry point is model_service.main.
"""

from control_plane.api import create_app
from control_plane.settings import Settings


app = create_app(Settings.from_env())
