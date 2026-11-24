"""The Ray Serve wrapper of the pipeline runner: `app` executes any resolved pipeline.

The chart deploys this one import path,
`model_service.pipeline_runner.serve:app`, as the `pipeline-runner`
application under `/pipelines` whenever a pipeline is deployed. Its
`user_config` holds only the replica's limits and allowed shared-storage
roots (`ReplicaSettings`); each request carries the resolved pipeline. The
wrapper only adapts Serve to `PipelineReplica`:

- `reconfigure(user_config)` builds a new replica core;
- `POST /`: one `PipelineRunRequest`, answered with the result or the error
  envelope of `contracts/pipeline_http.py`.

Models are reached through their Serve applications by ID
(`RayModel`).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from ray import serve

from model_service.contracts.pipeline_http import (
    ErrorCode,
    ErrorResponse,
    PipelineRunRequest,
)
from model_service.contracts.result import RunResult
from model_service.pipeline_runner.http_errors import (
    answer_http,
    error,
    invalid_request,
)
from model_service.pipeline_runner.logs import configure_replica_logging
from model_service.pipeline_runner.replica import PipelineReplica
from model_service.pipeline_runner.settings import ReplicaSettings


fastapi = FastAPI()
fastapi.add_exception_handler(RequestValidationError, invalid_request)  # type: ignore[arg-type]


@serve.deployment(name="PipelineRunner")
@serve.ingress(fastapi)
class PipelineRunnerDeployment:
    """One pipeline-runner replica, configured by its `user_config`."""

    def __init__(self) -> None:
        """Log model-service lines as JSON; the replica waits for config."""
        self._replica: PipelineReplica | None = None
        configure_replica_logging()

    def reconfigure(self, user_config: dict[str, Any]) -> None:
        """Build the replica core from new `user_config`.

        Args:
            user_config: `num_threads`, `max_concurrent_tasks` and
                `shared_storage_roots`, validated as `ReplicaSettings`.
        """
        self._replica = PipelineReplica(ReplicaSettings.model_validate(user_config))

    @fastapi.post(
        "/",
        response_model=RunResult,
        responses={code: {"model": ErrorResponse} for code in (422, 500, 503)},
    )
    async def run(self, body: PipelineRunRequest) -> RunResult:
        """Run one resolved pipeline over HTTP.

        Args:
            body: The resolved document, the invocation and the attempt.

        Returns:
            Each output's file and media type.

        Raises:
            HTTPException: 503 before the first reconfigure, or the error
                envelope `answer_http` maps the failure to.
        """
        if self._replica is None:
            raise error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                ErrorCode.UNAVAILABLE,
                "pipeline runner is not configured yet",
                retryable=True,
            )
        return await answer_http(self._replica.run(body))


app = PipelineRunnerDeployment.bind()  # type: ignore[attr-defined]
