from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from control_plane.domain import (
    AdmissionRequest,
    AdmissionResult,
    Execution,
)


if TYPE_CHECKING:
    from control_plane.registry import AppRegistry
    from control_plane.store import SQLiteExecutionStore


class UnknownAppMappingError(ValueError):
    pass


class InstallationAliasError(ValueError):
    pass


class ExecutionService:
    def __init__(
        self,
        store: SQLiteExecutionStore,
        registry: AppRegistry,
        installation_alias: str,
    ) -> None:
        self._store = store
        self._registry = registry
        self._installation_alias = installation_alias

    def admit(self, request: AdmissionRequest) -> AdmissionResult:
        if request.installation_alias != self._installation_alias:
            raise InstallationAliasError("installation alias is not configured")

        fingerprint = self._fingerprint(request)
        existing = self._store.find_by_submission(request, fingerprint)
        if existing is not None:
            return AdmissionResult(existing, created=False)

        mapping = self._registry.get(
            request.registered_app_id, request.registered_app_version
        )
        if mapping is None:
            raise UnknownAppMappingError("registered app mapping is not enabled")
        return self._store.admit(request, fingerprint, mapping)

    def get(self, execution_id: str) -> Execution:
        return self._store.get(execution_id)

    def cancel(self, execution_id: str) -> Execution:
        return self._store.request_cancellation(execution_id)

    @staticmethod
    def _fingerprint(request: AdmissionRequest) -> str:
        canonical = json.dumps(
            {
                "empaia_job_id": request.empaia_job_id,
                "installation_alias": request.installation_alias,
                "registered_app_id": request.registered_app_id,
                "registered_app_version": request.registered_app_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(canonical).hexdigest()
