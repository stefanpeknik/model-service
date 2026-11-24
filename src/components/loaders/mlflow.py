"""The `mlflow` loader: an artifact from an MLflow tracking server."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer

from components import Loader, Options


class MlflowOptions(Options):
    """Options of the `mlflow` loader.

    Attributes:
        uri: The artifact's URI.
        sha256: The artifact's expected SHA-256, if pinned.
    """

    uri: str = Field(
        min_length=1,
        description=(
            "The artifact's URI, e.g. `mlflow-artifacts:/<experiment>/<run>/"
            "artifacts/<path>`; it names one run's file."
        ),
    )
    sha256: str | None = Field(
        None,
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "The artifact's SHA-256 (`sha256sum` of the file). A run's artifact "
            "can be overwritten under the same URI; with this set, a download "
            "with other bytes is refused and the pipeline runner checks it."
        ),
    )

    @model_serializer(mode="wrap")
    def _unpinned_keeps_its_digest(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        """Leave out an unset `sha256`.

        The model digest, and the TensorRT cache directory named after
        it, then stay what they were before the option existed.

        Args:
            handler: Pydantic's default serializer.

        Returns:
            The options, `sha256` only when set.
        """
        data: dict[str, Any] = handler(self)
        if data.get("sha256") is None:
            data.pop("sha256", None)
        return data


class Mlflow(Loader):
    """Downloads one artifact (e.g. an ONNX file) from MLflow.

    `loader: {use: mlflow, uri: "mlflow-artifacts:/65/<run>/artifacts/model.onnx"}`.
    The tracking server comes from the environment (`MLFLOW_TRACKING_URI`),
    which each deployment environment sets once. After the download the
    revision carries the artifact's SHA-256, so `describe()` names the bytes
    that run; `sha256:` pins them.
    """

    name = "mlflow"
    Options = MlflowOptions
    example_options: ClassVar[dict[str, Any]] = {
        "uri": "mlflow-artifacts:/0/0123456789abcdef/artifacts/model.onnx"
    }
    options: MlflowOptions
    _fetched_sha256: str | None = None

    def fetch(self, cache_dir: Path) -> Path:
        """Download the artifact and checksum it.

        Args:
            cache_dir: Where to download it.

        Returns:
            The downloaded file's path.

        Raises:
            ValueError: If `sha256` is pinned and the download differs.
        """
        import mlflow.artifacts

        path = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=self.options.uri, dst_path=str(cache_dir)
            )
        )
        digest = _sha256(path)
        if self.options.sha256 is not None and digest != self.options.sha256:
            raise ValueError(
                f"{self.options.uri} has sha256 {digest}, but the model "
                f"pins {self.options.sha256}"
            )
        self._fetched_sha256 = digest
        return path

    def revision(self) -> str:
        """Name the weights by URI and, once known, their SHA-256.

        Returns:
            `<uri>@sha256:<hex>` after `fetch` or when pinned, else the URI.
        """
        digest = self._fetched_sha256 or self.options.sha256
        return (
            self.options.uri
            if digest is None
            else f"{self.options.uri}@sha256:{digest}"
        )

    def expected_revision(self) -> str | None:
        """The revision a pinned `sha256` promises.

        Returns:
            `<uri>@sha256:<pinned hex>`, or None when not pinned.
        """
        if self.options.sha256 is None:
            return None
        return f"{self.options.uri}@sha256:{self.options.sha256}"


def _sha256(path: Path) -> str:
    """Checksum a file, or a directory's files by relative path and content.

    Args:
        path: A downloaded file or directory.

    Returns:
        The SHA-256 as hex.
    """
    digest = hashlib.sha256()
    files = (
        [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    )
    for file in files:
        if file != path:
            digest.update(str(file.relative_to(path)).encode() + b"\0")
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()
