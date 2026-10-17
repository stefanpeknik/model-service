"""EMPAIA apps: which builder and model deployment each app runs.

Each app is one file, `deploy/helm/rayservice/empaia-apps/<app>.yaml`; the file
name is the app's ID. An app registered in EMPAIA as `http::<target>/<app>`
arrives at `POST /v1/empaia/apps/<app>`. Helm renders the files into one JSON
object, `{app_id: document}`, which the facade reads at startup. The same files
generate the apps' EADs, so routing and the EAD cannot disagree.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from model_service.builders.registry import get_builder


if TYPE_CHECKING:
    from pathlib import Path


class UnknownAppError(KeyError):
    pass


class EmpaiaApp(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    builder: Annotated[str, Field(min_length=1)]
    model: Annotated[str, Field(min_length=1)]


class EmpaiaApps:
    def __init__(self, apps: dict[str, EmpaiaApp]) -> None:
        for app in apps.values():
            get_builder(app.builder)
        self._apps = dict(apps)

    @classmethod
    def from_file(cls, path: Path) -> EmpaiaApps:
        try:
            documents = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(documents, dict):
                raise TypeError("EMPAIA apps must be an object keyed by app ID")
            apps = {
                app_id: EmpaiaApp.model_validate(document)
                for app_id, document in documents.items()
            }
        except (OSError, TypeError, ValueError, ValidationError) as error:
            raise ValueError(f"cannot read EMPAIA apps from {path}: {error}") from error
        return cls(apps)

    def get(self, app_id: str) -> EmpaiaApp:
        try:
            return self._apps[app_id]
        except KeyError as error:
            raise UnknownAppError(app_id) from error
