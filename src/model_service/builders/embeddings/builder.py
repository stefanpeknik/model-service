"""Tile-embeddings builder: an HDF5 feature table from any `tile-embedding` model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.builders.application import Builder
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


class EmbeddingBuilder(Builder[TileEmbeddingModel, EmbeddingConfiguration]):
    def __init__(
        self,
        settings: WorkerSettings,
        *,
        predictor: Callable[[TileEmbeddingModel], EmbeddingPredictor] = (
            RayEmbeddingPredictor
        ),
    ) -> None:
        """Configure the builder for a replica.

        Args:
            settings: Replica settings for thread and concurrency limits.
            predictor: Factory creating a predictor for a model; replaceable in tests.
        """
        self._settings = settings
        self._predictor = predictor

    async def run(
        self, context: BuilderContext[TileEmbeddingModel, EmbeddingConfiguration]
    ) -> BuilderResult:
        """Write a slide's tile embeddings to an HDF5 artifact.

        Checks the running model against its catalog entry first, then writes
        through a temporary file so a failed run leaves no partial artifact.

        Args:
            context: The run's inputs, model and configuration.

        Returns:
            The `embeddings` artifact.

        Raises:
            BuilderFailure: The model's configuration differs from its catalog
                entry, a model request fails, or the model returns invalid features.
            InvalidInvocation: The slide or region of interest is unreadable, or the
                output would replace an input.
        """
        model = context.model
        config = context.configuration
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
            outputs={"embeddings": SharedArtifact(path=str(destination))}
        )
