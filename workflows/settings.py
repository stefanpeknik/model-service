from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class WorkflowWorkerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    num_threads: int = Field(default=2, gt=0)
    max_concurrent_tasks: int = Field(default=4, gt=0)
    model_origin: HttpUrl = HttpUrl("http://127.0.0.1:8000")
