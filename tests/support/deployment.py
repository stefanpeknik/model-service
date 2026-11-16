"""The Helm chart's values files, read as the chart reads them."""

from __future__ import annotations

from pathlib import Path

import yaml


CHART = Path(__file__).parents[2] / "deploy/helm/rayservice"
VALUES = sorted(CHART.glob("values*.yaml"))


def load(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))
