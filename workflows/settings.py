"""Deployment-only limits and internal raw-model origin for workflow replicas."""

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class WorkflowWorkerSettings(BaseModel):
    """Bound concurrent slide processing independently of per-execution configuration.

    Attributes:
        num_threads: Maximum synchronous slide-read threads per invocation.
        max_concurrent_tasks: Maximum tiles being read or predicted per invocation.
        model_origin: Internal Ray Serve origin hosting the unchanged raw model routes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    num_threads: int = Field(default=2, gt=0)
    max_concurrent_tasks: int = Field(default=4, gt=0)
    model_origin: HttpUrl = HttpUrl("http://127.0.0.1:8000")
