"""Tile-embeddings builder: an HDF5 feature table from any `tile-embedding` model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from builders.artifacts import atomic_artifact
from builders.contracts import BuilderResult, SharedArtifact, TileEmbeddingModel
from builders.embeddings.predictor import HttpEmbeddingPredictor
from builders.embeddings.processor import build_embeddings
from builders.registry import EmbeddingConfiguration


if TYPE_CHECKING:
    from builders.application import BuilderContext
    from builders.settings import WorkerSettings


class EmbeddingBuilder:
    def __init__(
        self,
        settings: WorkerSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    async def run(self, context: BuilderContext) -> BuilderResult:
        model = context.model
        config = context.configuration
        assert isinstance(model, TileEmbeddingModel)
        assert isinstance(config, EmbeddingConfiguration)
        slide = context.input_path("slide")
        mask = context.optional_input_path("region_of_interest")
        destination = context.output_path("embeddings")
        inputs = (slide,) if mask is None else (slide, mask)
        with atomic_artifact(destination, inputs=inputs) as temporary:
            async with httpx.AsyncClient(
                base_url=str(self._settings.model_origin),
                transport=self._transport,
                timeout=None,
                trust_env=False,
            ) as client:
                await build_embeddings(
                    slide,
                    mask,
                    temporary,
                    predictor=HttpEmbeddingPredictor(client, model),
                    model=model,
                    stride_fraction=config.stride_fraction,
                    num_threads=self._settings.num_threads,
                    max_concurrent_tasks=self._settings.max_concurrent_tasks,
                )
        return BuilderResult(
            outputs={
                "embeddings": SharedArtifact(
                    path=str(destination), media_type="application/x-hdf5"
                )
            }
        )
