"""Ray Serve ingress hosting one builder, selected by `builder_id` in user_config.

Every builder application uses this module; only its user_config differs.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from pydantic import Field
from ray import serve

from model_service.builder_deployment.http_handler import (
    error,
    invalid_request,
    run_http,
)
from model_service.builder_deployment.runner import BuilderApplication
from model_service.builders.base import ReplicaSettings
from model_service.builders.registry import get_builder
from model_service.contracts.builder_http import ErrorCode, ErrorResponse, RunRequest
from model_service.contracts.result import BuilderResult
from model_service.providers.registry import PROVIDERS


if TYPE_CHECKING:
    from model_service.builders.input_access import InputAccessFactory


fastapi = FastAPI()
fastapi.add_exception_handler(RequestValidationError, invalid_request)  # type: ignore[arg-type]


class DeploymentSettings(ReplicaSettings):
    """The deployment's Ray `user_config`: which builder, and its replica settings."""

    builder_id: str = Field(min_length=1)


def build_application(config: DeploymentSettings) -> BuilderApplication:
    """Load the configured builder and its providers' access adapters.

    Args:
        config: Deployment settings naming the builder and worker settings.

    Returns:
        The builder application the deployment serves.

    Raises:
        UnknownBuilderError: `builder_id` names no registered builder.
    """
    spec = get_builder(config.builder_id)
    module_name, _, class_name = spec.implementation.partition(":")
    implementation = getattr(importlib.import_module(module_name), class_name)
    settings = ReplicaSettings.model_validate(config.model_dump(exclude={"builder_id"}))
    adapters: dict[str, InputAccessFactory] = {}
    for provider in PROVIDERS:
        adapters.update(provider.input_access(settings))
    return BuilderApplication(spec, implementation(settings), adapters)


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class BuilderDeployment:
    def __init__(self) -> None:
        """Route model-service logs to Serve's handlers; the builder waits for config."""
        self._application: BuilderApplication | None = None
        # Serve only sets up its own logger; without this, model-service INFO logs
        # never reach the replica log or the dashboard.
        package_logger = logging.getLogger("model_service")
        package_logger.handlers = logging.getLogger("ray.serve").handlers
        package_logger.setLevel(logging.INFO)
        package_logger.propagate = False

    def reconfigure(self, values: dict[str, object]) -> None:
        """Rebuild the builder application from new user_config.

        Args:
            values: Ray Serve user_config, validated as `DeploymentSettings`.
        """
        self._application = build_application(DeploymentSettings.model_validate(values))

    @fastapi.post(
        "/",
        response_model=BuilderResult,
        responses={code: {"model": ErrorResponse} for code in (422, 500, 503)},
    )
    async def run(self, body: RunRequest) -> BuilderResult:
        """Run one builder invocation over HTTP.

        Args:
            body: Invocation, model and execution context of the attempt.

        Returns:
            The builder's result.

        Raises:
            HTTPException: 503 before the first reconfigure, or the error envelope
                from `run_http`.
        """
        if self._application is None:
            raise error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                ErrorCode.UNAVAILABLE,
                "builder is not configured yet",
                retryable=True,
            )
        return await run_http(self._application, body)


app = BuilderDeployment.bind()  # type: ignore[attr-defined]
