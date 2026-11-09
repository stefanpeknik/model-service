"""Tile-embeddings builder: an HDF5 feature table from any `tile-embedding` model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.builders.base import Builder
from model_service.builders.embeddings.declaration import (
    EmbeddingConfiguration,
    EmbeddingInputs,
    EmbeddingOutputs,
)
from model_service.builders.embeddings.processor import build_embeddings
from model_service.builders.model_clients import TileEmbeddingClient


if TYPE_CHECKING:
    from model_service.builders.base import BuilderContext
    from model_service.builders.settings import WorkerSettings


class EmbeddingBuilder(
    Builder[
        TileEmbeddingClient, EmbeddingInputs, EmbeddingConfiguration, EmbeddingOutputs
    ]
):
    def __init__(self, settings: WorkerSettings) -> None:
        """Configure the builder for a replica.

        Args:
            settings: Replica settings for thread and concurrency limits.
        """
        self._settings = settings

    async def run(
        self,
        context: BuilderContext[
            TileEmbeddingClient, EmbeddingInputs, EmbeddingConfiguration
        ],
    ) -> EmbeddingOutputs:
        """Write a slide's tile embeddings to an HDF5 artifact.

        Args:
            context: The verified model, the slide and optional region of interest,
                and the configuration.

        Returns:
            The `embeddings` artifact.

        Raises:
            BuilderFailure: A model call failed or returned invalid features.
            InvalidInvocation: The slide or region of interest is unreadable, or the
                output would replace an input.
        """
        with context.write_artifact("embeddings") as embeddings:
            await build_embeddings(
                context.inputs.slide,
                context.inputs.region_of_interest,
                embeddings.path,
                client=context.model,
                stride_fraction=context.configuration.stride_fraction,
                num_threads=self._settings.num_threads,
                max_concurrent_tasks=self._settings.max_concurrent_tasks,
            )
        return EmbeddingOutputs(embeddings=embeddings.artifact)
