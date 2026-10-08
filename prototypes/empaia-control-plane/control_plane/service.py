"""Validate prototype installation/app admission and delegate durable lifecycle operations.

Historical SQLite control-plane prototype. The current production scheduler lives in
execution_scheduler/ and uses PostgreSQL; these lifecycle rules are not its
specification.
"""

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
    """Signal an app/version pair not enabled by deployment configuration."""


class InstallationAliasError(ValueError):
    """Signal admission targeted at a different configured installation."""


class ExecutionService:
    """Coordinate replay lookup and allow-listed app admission over the SQLite store."""

    def __init__(
        self,
        store: SQLiteExecutionStore,
        registry: AppRegistry,
        installation_alias: str,
    ) -> None:
        """Retain caller-owned storage/registry and the permitted installation alias.

        Args:
            store: Historical prototype transactional SQLite execution store.
            registry: Allow-list of enabled prototype app/version mappings.
            installation_alias: Deployment-owned prototype installation identity
                permitted for admission.
        """
        self._store = store
        self._registry = registry
        self._installation_alias = installation_alias

    def admit(self, request: AdmissionRequest) -> AdmissionResult:
        """Validate the installation, replay existing identity before registry lookup, and pin the enabled app mapping on new admission.

        Args:
            request: Prototype installation/job/app identity and replay key to admit.

        Returns:
            AdmissionResult: The admitted execution and whether this operation created
                it.

        Raises:
            InstallationAliasError: The admission targets an unconfigured prototype
                installation.
            UnknownAppMappingError: The prototype app/version mapping is not enabled.
        """
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
        """Read the prototype durable execution by ID.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            Execution: Historical prototype execution snapshot with its pinned app
                contract and lifecycle.
        """
        return self._store.get(execution_id)

    def cancel(self, execution_id: str) -> Execution:
        """Record cancellation through the prototype store.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            Execution: Historical prototype execution snapshot with its pinned app
                contract and lifecycle.
        """
        return self._store.request_cancellation(execution_id)

    @staticmethod
    def _fingerprint(request: AdmissionRequest) -> str:
        """Hash canonical installation, job, and app/version fields for conflict detection.

        Args:
            request: Prototype installation/job/app identity and replay key to admit.

        Returns:
            str: The identifier, path, or normalized text described above.
        """
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
