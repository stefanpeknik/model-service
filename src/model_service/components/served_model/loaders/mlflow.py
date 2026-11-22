"""The `mlflow` loader: an artifact from an MLflow tracking server."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from model_service.components.kinds import Loader
from model_service.components.options import Options


class MlflowOptions(Options):
    """Options of the `mlflow` loader.

    Attributes:
        uri: The artifact's URI.
    """

    uri: str = Field(
        min_length=1,
        description=(
            "The artifact's URI, e.g. `mlflow-artifacts:/<experiment>/<run>/"
            "artifacts/<path>`; it names one run's file, so it pins the weights."
        ),
    )


class Mlflow(Loader):
    """Downloads one artifact (e.g. an ONNX file) from MLflow.

    `weights: {use: mlflow, uri: "mlflow-artifacts:/65/<run>/artifacts/model.onnx"}`.
    The tracking server comes from the environment (`MLFLOW_TRACKING_URI`),
    which each deployment environment sets once.
    """

    name = "mlflow"
    Options = MlflowOptions
    example_options: ClassVar[dict[str, Any]] = {
        "uri": "mlflow-artifacts:/0/0123456789abcdef/artifacts/model.onnx"
    }
    options: MlflowOptions

    def fetch(self, cache_dir: Path) -> Path:
        """Download the artifact.

        Args:
            cache_dir: Where to download it.

        Returns:
            The downloaded file's path.
        """
        import mlflow.artifacts

        return Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=self.options.uri, dst_path=str(cache_dir)
            )
        )

    def revision(self) -> str:
        """Name the weights by their artifact URI, which includes the run.

        Returns:
            The artifact URI.
        """
        return self.options.uri
