"""The `huggingface` loader: a model repository on the Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from model_service.components.kinds import Loader
from model_service.components.options import Options


if TYPE_CHECKING:
    from model_service.components.problems import Problem


class HuggingfaceOptions(Options):
    """Options of the `huggingface` loader.

    Attributes:
        repo: The repository ID.
        revision: The commit to download.
        files: Glob patterns of the files to download, or None for all.
    """

    repo: str = Field(
        pattern=r"^[\w.-]+/[\w.-]+$",
        description="The repository ID, `<owner>/<name>`, e.g. `paige-ai/Virchow2`.",
    )
    revision: str = Field(
        pattern=r"^[0-9a-f]{40}$",
        description=(
            "The full 40-character commit hash to download. A branch or tag can "
            "move, so the weights would change under the same file."
        ),
    )
    files: tuple[str, ...] | None = Field(
        None,
        min_length=1,
        description=(
            "Glob patterns of the files to download, e.g. "
            "`[config.json, model.safetensors]`; left out, the whole repository."
        ),
    )


class Huggingface(Loader):
    """Downloads one commit of a Hugging Face Hub repository.

    `weights: {use: huggingface, repo: paige-ai/Virchow2, revision: <commit>}`.
    The files go to the Hub's own cache (`HF_HOME`), which each deployment
    environment points at a shared volume, so a restart downloads nothing.
    A gated repository needs `HF_TOKEN` in the environment.
    """

    name = "huggingface"
    Options = HuggingfaceOptions
    example_options: ClassVar[dict[str, Any]] = {
        "repo": "paige-ai/Virchow2",
        "revision": "3158645804b69e3f3bc4439d4116edddf0840a72",
    }
    options: HuggingfaceOptions

    def fetch(self, cache_dir: Path) -> Path:
        """Download the commit, or find it in the Hub's cache.

        Args:
            cache_dir: Unused; the Hub's cache outlives the replica.

        Returns:
            The local directory holding the commit's files.
        """
        from huggingface_hub import snapshot_download

        files = list(self.options.files) if self.options.files else None
        return Path(
            snapshot_download(
                self.options.repo, revision=self.options.revision, allow_patterns=files
            )
        )

    def revision(self) -> str:
        """Name the weights by repository and commit.

        Returns:
            `<repo>@<commit>`.
        """
        return f"{self.options.repo}@{self.options.revision}"

    def expected_revision(self) -> str:
        """The commit is pinned, so the revision is known before fetching.

        Returns:
            `<repo>@<commit>`.
        """
        return self.revision()

    @classmethod
    def hint(cls, problem: Problem) -> str | None:
        """Point a revision that is not a commit hash at the repository's history.

        Args:
            problem: The failed check.

        Returns:
            The hint for `revision`, else None.
        """
        if problem.detail.get("option") == "revision":
            return (
                "copy the full commit hash from the history of the "
                "repository's files on huggingface.co"
            )
        return None
