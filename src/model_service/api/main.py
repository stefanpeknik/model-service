"""The production FastAPI application Uvicorn imports.

Uvicorn configures logging before importing it, from `log-config.json` next to
this module (`--log-config`): JSON lines on stdout, each with its request id.
"""

from model_service.api.app_factory import build_api_process


app = build_api_process()
