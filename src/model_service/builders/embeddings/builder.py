"""Tile-embeddings builder: an HDF5 feature table from any `tile-embedding` model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.builders.artifacts import atomic_artifact
from model_service.builders.contracts import (
    BuilderResult,
    SharedArtifact,
    TileEmbeddingModel,
)
from model_service.builders.embeddings.predictor import RayEmbeddingPredictor
from model_service.builders.embeddings.processor import build_embeddings
from model_service.builders.registry import EmbeddingConfiguration


if TYPE_CHECKING:
    from collections.abc import Callable

    from model_service.builders.application import BuilderContext
    from model_service.builders.embeddings.predictor import EmbeddingPredictor
    from model_service.builders.settings import WorkerSettings


class EmbeddingBuilder:
    def __init__(
        self,
        settings: WorkerSettings,
        *,
        predictor: Callable[[TileEmbeddingModel], EmbeddingPredictor] = (
            RayEmbeddingPredictor
        ),
    ) -> None:
        self._settings = settings
        self._predictor = predictor

    async def run(self, context: BuilderContext) -> BuilderResult:
        model = context.model
        config = context.configuration
        assert isinstance(model, TileEmbeddingModel)
        assert isinstance(config, EmbeddingConfiguration)
        predictor = self._predictor(model)
        await predictor.verify()
        slide = context.input_path("slide")
        mask = context.optional_input_path("region_of_interest")
        destination = context.output_path("embeddings")
        inputs = (slide,) if mask is None else (slide, mask)
        with atomic_artifact(destination, inputs=inputs) as temporary:
            await build_embeddings(
                slide,
                mask,
                temporary,
                predictor=predictor,
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
