"""The plain (non-component) blocks of a served-model file."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Block(BaseModel):
    """A block of a served-model file: unknown keys are errors."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Input(Block):
    """What the model takes: RGB uint8 tiles, `(3, tile_size, tile_size)` per tile.

    Attributes:
        tile_size: Tile edge in pixels.
        mpp: Microns per pixel the model was trained at.
    """

    tile_size: int = Field(gt=0, description="Tile edge in pixels.")
    mpp: float = Field(
        gt=0, allow_inf_nan=False, description="Microns per pixel of the tiles."
    )


class ScoresOutput(Block):
    """A score map per tile: `(channels, size, size)`.

    Attributes:
        kind: `scores`.
        channels: Score channels.
        size: Edge of the score map; 1 for one score per tile.
        dtype: The scores' NumPy dtype.
        range: The interval every score must lie in, if declared.
    """

    kind: Literal["scores"] = Field(description="`scores` or `embedding`.")
    channels: int = Field(1, gt=0, description="Score channels.")
    size: int = Field(
        1, gt=0, description="Edge of the score map; 1 for one score per tile."
    )
    dtype: Literal["float32", "float16"] = Field(
        "float32", description="The scores' dtype: `float32` or `float16`."
    )
    range: tuple[float, float] | None = Field(
        None,
        description=(
            "`[low, high]` every score must lie in, e.g. `[0, 1]` for "
            "probabilities; a model returning logits then fails on its first batch."
        ),
    )

    @model_validator(mode="after")
    def _ordered_range(self) -> ScoresOutput:
        if self.range is not None and self.range[0] > self.range[1]:
            raise ValueError("range must be [low, high] with low <= high")
        return self


class EmbeddingOutput(Block):
    """One feature vector per summary for each tile.

    Attributes:
        kind: `embedding`.
        summaries: The summary names; one vector each, in this order.
        dtype: The features' NumPy dtype.
    """

    kind: Literal["embedding"] = Field(description="`scores` or `embedding`.")
    summaries: tuple[Annotated[str, Field(min_length=1)], ...] = Field(
        min_length=1,
        description=(
            "Summary names, one vector each, e.g. `[global, local_mean]`; with "
            "one summary a tile's output is one vector."
        ),
    )
    dtype: Literal["float32"] = Field("float32", description="`float32`.")


type Output = Annotated[ScoresOutput | EmbeddingOutput, Field(discriminator="kind")]


class Batching(Block):
    """How requests are batched for the runtime.

    Attributes:
        max_size: The largest batch.
        wait_s: How long a batch waits to fill.
    """

    max_size: int = Field(gt=0, description="The largest batch the runtime gets.")
    wait_s: float = Field(ge=0, description="Seconds a batch waits to fill up.")


class Resources(Block):
    """What one replica reserves.

    Attributes:
        gpus: GPUs (fractions share one).
        cpus: CPUs.
        memory_gib: Memory in GiB.
    """

    gpus: float = Field(0, ge=0, description="GPUs per replica; 0.5 shares one.")
    cpus: float = Field(1, gt=0, description="CPUs per replica.")
    memory_gib: float = Field(gt=0, description="Memory per replica in GiB.")


class Serving(Block):
    """Scaling and queue limits; left out, they follow from `batching.max_size`.

    Attributes:
        min_replicas: Replicas kept when idle.
        max_replicas: Most replicas.
        target_ongoing_requests: Requests per replica the autoscaler aims at.
        max_ongoing_requests: Requests one replica takes at once.
        max_queued_requests: Requests queued before callers are refused.
        downscale_delay_s: Idle seconds before the autoscaler removes a replica.
    """

    min_replicas: int = Field(0, ge=0, description="Replicas kept when idle.")
    max_replicas: int = Field(4, ge=1, description="The most replicas.")
    target_ongoing_requests: int | None = Field(
        None,
        gt=0,
        description="Requests per replica the autoscaler aims at; half a batch.",
    )
    max_ongoing_requests: int | None = Field(
        None, gt=0, description="Requests one replica takes at once; two batches."
    )
    max_queued_requests: int | None = Field(
        None,
        gt=0,
        description="Requests queued before callers are refused; sixteen batches.",
    )
    downscale_delay_s: float | None = Field(
        None,
        ge=0,
        description=(
            "Seconds of low load before the autoscaler removes a replica; Ray "
            "Serve's default (600) when left out."
        ),
    )


class Fields(Block):
    """Every top-level key of a served-model file.

    The component keys are resolved through the registry; here they are only
    listed so a misspelt key names every key that fits.

    Attributes:
        weights: The loader.
        runtime: The runtime.
        pre: Steps before the runtime, per tile.
        post: Steps after the runtime, per tile.
        input: What the model takes.
        output: What the model returns.
        batching: How requests are batched.
        resources: What a replica reserves.
        serving: Scaling and queue limits.
    """

    weights: Any = Field(description="A loader: `{use: mlflow, uri: …}`.")
    runtime: Any = Field(description="A runtime: `{use: onnx, …}`.")
    pre: Any = Field(None, description="Steps run on each tile before the runtime.")
    post: Any = Field(None, description="Steps run on each output after the runtime.")
    input: Input = Field(description="`{tile_size: …, mpp: …}`.")
    output: Output = Field(
        description="`{kind: scores, …}` or `{kind: embedding, summaries: […]}`."
    )
    batching: Batching = Field(description="`{max_size: …, wait_s: …}`.")
    resources: Resources = Field(description="`{gpus: …, cpus: …, memory_gib: …}`.")
    serving: Serving = Field(
        default_factory=Serving, description="Scaling and queue limits."
    )
