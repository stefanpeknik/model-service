"""The Ray Serve wrapper of the analysis runner: `app` executes any resolved analysis.

The chart deploys this one import path,
`model_service.analysis_runner.serve:app`, as the `analysis-runner`
application under `/analyses` whenever an analysis is deployed. Its
`user_config` holds only the replica's limits and allowed shared-storage
roots (`ReplicaSettings`); each request carries the resolved analysis. The
wrapper only adapts Serve to `AnalysisReplica`:

- `reconfigure(user_config)` builds a new replica core;
- `POST /`: one `AnalysisRunRequest`, answered with the result or the error
  envelope of `contracts/analysis_http.py`.

Served models are reached through their Serve applications by ID
(`RayServedModel`).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from ray import serve

from model_service.analysis_runner.http_errors import (
    answer_http,
    error,
    invalid_request,
)
from model_service.analysis_runner.logs import configure_replica_logging
from model_service.analysis_runner.replica import AnalysisReplica
from model_service.analysis_runner.settings import ReplicaSettings
from model_service.contracts.analysis_http import (
    AnalysisRunRequest,
    ErrorCode,
    ErrorResponse,
)
from model_service.contracts.result import RunResult


fastapi = FastAPI()
fastapi.add_exception_handler(RequestValidationError, invalid_request)  # type: ignore[arg-type]


@serve.deployment(name="AnalysisRunner")
@serve.ingress(fastapi)
class AnalysisRunnerDeployment:
    """One analysis-runner replica, configured by its `user_config`."""

    def __init__(self) -> None:
        """Log model-service lines as JSON; the replica waits for config."""
        self._replica: AnalysisReplica | None = None
        configure_replica_logging()

    def reconfigure(self, user_config: dict[str, Any]) -> None:
        """Build the replica core from new `user_config`.

        Args:
            user_config: `num_threads`, `max_concurrent_tasks` and
                `shared_storage_roots`, validated as `ReplicaSettings`.
        """
        self._replica = AnalysisReplica(ReplicaSettings.model_validate(user_config))

    @fastapi.post(
        "/",
        response_model=RunResult,
        responses={code: {"model": ErrorResponse} for code in (422, 500, 503)},
    )
    async def run(self, body: AnalysisRunRequest) -> RunResult:
        """Run one resolved analysis over HTTP.

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
                "analysis runner is not configured yet",
                retryable=True,
            )
        return await answer_http(self._replica.run(body))


app = AnalysisRunnerDeployment.bind()  # type: ignore[attr-defined]
