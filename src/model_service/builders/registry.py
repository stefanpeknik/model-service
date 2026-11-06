"""Declare the available builders.

Declarations import only Pydantic so the facade can validate requests and list
builders without the slide-processing dependencies. Implementations are loaded
lazily from `implementation` inside Ray replicas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from model_service.builders.contracts import (
    Artifact,
    BuilderOutputs,
    InferenceContract,
    InlineOutput,
    InvalidInvocation,
    OutputKind,
    SharedArtifact,
    UnknownBuilderError,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.builders.contracts import BuilderResult, JsonValue


@dataclass(frozen=True, slots=True)
class InputSpec:
    required: bool = True
    media_type: str | None = None


@dataclass(frozen=True, slots=True)
class OutputSpec:
    kind: OutputKind
    media_type: str | None = None
    default_path: str | None = None


def output_specs(outputs: type[BuilderOutputs]) -> dict[str, OutputSpec]:
    """Derive an output declaration from a builder's outputs model.

    Args:
        outputs: The builder's `BuilderOutputs` subclass.

    Returns:
        One `OutputSpec` per field, in field order.

    Raises:
        TypeError: A field is neither a `SharedArtifact` nor an `InlineOutput`.
    """
    specs = {}
    for name, info in outputs.model_fields.items():
        if info.annotation is InlineOutput:
            specs[name] = OutputSpec(kind=OutputKind.INLINE)
        elif info.annotation is SharedArtifact:
            artifact = next(
                (m for m in info.metadata if isinstance(m, Artifact)), Artifact()
            )
            specs[name] = OutputSpec(
                kind=OutputKind.SHARED_ARTIFACT,
                media_type=artifact.media_type,
                default_path=artifact.default_path,
            )
        else:
            raise TypeError(f"{outputs.__name__}.{name} is not a builder output")
    return specs


@dataclass(frozen=True, slots=True)
class BuilderSpec:
    id: str
    accepts: InferenceContract
    inputs: Mapping[str, InputSpec]
    # What the builder returns; `outputs` is derived from it.
    result: type[BuilderOutputs]
    configuration: type[BaseModel]
    implementation: str
    outputs: Mapping[str, OutputSpec] = field(init=False)

    def __post_init__(self) -> None:
        """Derive the output declaration from the outputs model."""
        object.__setattr__(self, "outputs", output_specs(self.result))

    def resolve_configuration(
        self, overrides: Mapping[str, JsonValue] | None = None
    ) -> BaseModel:
        """Apply caller or provider overrides to the builder's current defaults.

        Args:
            overrides: Configuration fields to override; none uses the defaults.

        Returns:
            The validated configuration.

        Raises:
            InvalidInvocation: The overrides are invalid; the message names the
                fields only, since values may come from callers.
        """
        try:
            return self.configuration.model_validate(dict(overrides or {}))
        except ValidationError as error:
            # Field names only: values may come from callers.
            fields = sorted({".".join(map(str, e["loc"])) for e in error.errors()})
            raise InvalidInvocation(
                f"{self.id} configuration is invalid: {', '.join(fields)}"
            ) from error


_STRIDE = Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]


class HeatmapConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stride_fraction: _STRIDE = 0.5
    # Skip tiles without tissue; they score 0.
    skip_background: bool = True


class EmbeddingConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stride_fraction: _STRIDE = 1.0


class HeatmapOutputs(BuilderOutputs):
    heatmap: Annotated[
        SharedArtifact,
        Artifact(media_type="application/x-npz", default_path="heatmap.npz"),
    ]


class EmbeddingOutputs(BuilderOutputs):
    embeddings: Annotated[
        SharedArtifact,
        Artifact(media_type="application/x-hdf5", default_path="embeddings.h5"),
    ]


_SLIDE = InputSpec(media_type="application/vnd.openmicroscopy")
_REGION_OF_INTEREST = InputSpec(required=False, media_type="image/tiff")

BUILDERS: dict[str, BuilderSpec] = {
    spec.id: spec
    for spec in (
        BuilderSpec(
            id="heatmap",
            accepts=InferenceContract.SPATIAL_SCORES,
            inputs={"slide": _SLIDE, "region_of_interest": _REGION_OF_INTEREST},
            result=HeatmapOutputs,
            configuration=HeatmapConfiguration,
            implementation="model_service.builders.heatmap.builder:HeatmapBuilder",
        ),
        BuilderSpec(
            id="tile-embeddings",
            accepts=InferenceContract.TILE_EMBEDDING,
            inputs={"slide": _SLIDE, "region_of_interest": _REGION_OF_INTEREST},
            result=EmbeddingOutputs,
            configuration=EmbeddingConfiguration,
            implementation="model_service.builders.embeddings.builder:EmbeddingBuilder",
        ),
    )
}


def get_builder(builder_id: str) -> BuilderSpec:
    """Look up a builder declaration by id.

    Args:
        builder_id: Registered builder id.

    Returns:
        The builder's declaration.

    Raises:
        UnknownBuilderError: No builder has that id.
    """
    try:
        return BUILDERS[builder_id]
    except KeyError as error:
        raise UnknownBuilderError(builder_id) from error


def parse_result(builder_id: str, result: BuilderResult) -> BuilderOutputs:
    """Read a stored or received result as its builder's outputs model.

    Args:
        builder_id: The builder that produced the result.
        result: The generic result, as sent and stored.

    Returns:
        The result as the builder's `BuilderOutputs` subclass.

    Raises:
        UnknownBuilderError: No builder has this id.
        ValidationError: The result does not match the builder's current
            outputs, e.g. an output was renamed since it was stored.
    """
    return get_builder(builder_id).result.model_validate(result.outputs)
