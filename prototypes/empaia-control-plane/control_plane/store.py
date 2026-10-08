"""Persist the prototype state machine through operation-scoped SQLite transactions.

Historical SQLite control-plane prototype. The current production scheduler lives in
execution_scheduler/ and uses PostgreSQL; these lifecycle rules are not its
specification.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from control_plane.domain import (
    AdmissionRequest,
    AdmissionResult,
    AppMapping,
    AttemptOwnershipError,
    Execution,
    ExecutionNotFoundError,
    ExecutionSpec,
    ExecutionState,
    FailureCategory,
    IncompatibleSchemaError,
    InvalidTransitionError,
    SubmissionConflictError,
    TimeoutKind,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


_SCHEMA_VERSION = 1
_EARLY_CANCELLATION_STATES = {
    ExecutionState.ACCEPTED,
    ExecutionState.RESOLVING,
    ExecutionState.READY,
    ExecutionState.OUTPUT_READY,
}
_STATE_VALUES = ", ".join(f"'{state.value}'" for state in ExecutionState)
_FAILURE_CATEGORY_VALUES = ", ".join(
    f"'{category.value}'" for category in FailureCategory
)
_TIMEOUT_KIND_VALUES = ", ".join(f"'{kind.value}'" for kind in TimeoutKind)


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for the default prototype store clock.

    Returns:
        datetime: Current timezone-aware UTC wall-clock timestamp.
    """
    return datetime.now(UTC)


class SQLiteExecutionStore:
    """Transactional execution store for one control-plane process."""

    def __init__(
        self,
        database_path: Path,
        clock: Callable[[], datetime] = utc_now,
        busy_timeout_seconds: float = 2.0,
    ) -> None:
        """Create a private database file and initialize its versioned schema using the configured clock and busy timeout.

        Args:
            database_path: Filesystem location of the historical prototype SQLite
                database.
            clock: Aware timestamp source for admission or deterministic tests; defaults
                to UTC wall time.
            busy_timeout_seconds: Maximum time SQLite waits for a competing database
                lock.

        Raises:
            ValueError: A value fails the validation described above.
        """
        if busy_timeout_seconds <= 0:
            raise ValueError("busy_timeout_seconds must be positive")
        self._database_path = database_path
        self._clock = clock
        self._busy_timeout_seconds = busy_timeout_seconds
        database_path.parent.mkdir(parents=True, exist_ok=True)
        database_path.touch(mode=0o600, exist_ok=True)
        os.chmod(database_path, 0o600)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """Open an operation-scoped connection with row access, foreign keys, and the busy timeout enabled.

        Returns:
            sqlite3.Connection: Open SQLite connection with prototype row and timeout
                settings.
        """
        connection = sqlite3.connect(
            self._database_path,
            timeout=self._busy_timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            f"PRAGMA busy_timeout = {round(self._busy_timeout_seconds * 1000)}"
        )
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection for reads and close it on every exit.

        Yields:
            sqlite3.Connection: Connection owned and closed by this context.
        """
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self, mode: str = "IMMEDIATE") -> Iterator[sqlite3.Connection]:
        """Begin the selected transaction mode, commit on success, roll back on failure, and always close the connection.

        Args:
            mode: SQLite transaction locking mode, normally IMMEDIATE.

        Yields:
            sqlite3.Connection: Connection owned and closed by this context.
        """
        connection = self._connect()
        try:
            connection.execute(f"BEGIN {mode}")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        """Create schema version one or reject incompatible/unversioned databases, then enable WAL journaling.

        Returns:
            None: No return value.

        Raises:
            IncompatibleSchemaError: The stored prototype schema or JSON data is
                unsupported.
        """
        with self._transaction(mode="EXCLUSIVE") as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > _SCHEMA_VERSION:
                raise IncompatibleSchemaError(
                    f"database schema {version} is newer than supported "
                    f"schema {_SCHEMA_VERSION}"
                )
            if version == 0:
                existing = connection.execute(
                    """
                    SELECT 1 FROM sqlite_master
                    WHERE type = 'table' AND name = 'executions'
                    """
                ).fetchone()
                if existing is not None:
                    raise IncompatibleSchemaError(
                        "unversioned executions table requires explicit migration"
                    )
                self._create_schema_v1(connection)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")

    @staticmethod
    def _create_schema_v1(connection: sqlite3.Connection) -> None:
        """Create the prototype execution table and indexes with replay, ownership, counter, and lifecycle constraints.

        Args:
            connection: Database connection owned by the enclosing operation or
                transaction.

        Returns:
            None: No return value.
        """
        connection.execute(
            f"""
            CREATE TABLE executions (
                id TEXT PRIMARY KEY,
                installation_alias TEXT NOT NULL,
                empaia_job_id TEXT NOT NULL,
                registered_app_id TEXT NOT NULL,
                registered_app_version TEXT NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE,
                request_fingerprint TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                workflow TEXT NOT NULL,
                resource_profile TEXT NOT NULL,
                max_runtime_seconds INTEGER NOT NULL CHECK (max_runtime_seconds > 0),
                max_attempts INTEGER NOT NULL CHECK (max_attempts > 0),
                input_name TEXT NOT NULL,
                output_name TEXT NOT NULL,
                model_parameter_names_json TEXT NOT NULL,
                allowed_wsi_plugin_ids_json TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ({_STATE_VALUES})),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                resolution_after TEXT,
                ready_at TEXT,
                first_claimed_at TEXT,
                current_attempt_id TEXT,
                attempt_heartbeat_at TEXT,
                dispatch_after TEXT,
                attempt_ids_json TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
                cancellation_requested INTEGER NOT NULL DEFAULT 0
                    CHECK (cancellation_requested IN (0, 1)),
                failure_category TEXT CHECK (
                    failure_category IS NULL OR
                    failure_category IN ({_FAILURE_CATEGORY_VALUES})
                ),
                timeout_kind TEXT CHECK (
                    timeout_kind IS NULL OR timeout_kind IN ({_TIMEOUT_KIND_VALUES})
                ),
                safe_reason_code TEXT,
                safe_reason_message TEXT,
                execution_spec_json TEXT,
                UNIQUE (installation_alias, empaia_job_id),
                CHECK ((safe_reason_code IS NULL) = (safe_reason_message IS NULL)),
                CHECK (
                    failure_category IS NULL OR state IN ('RESOLVING', 'FAILED')
                ),
                CHECK (timeout_kind IS NULL OR state = 'TIMED_OUT'),
                CHECK (
                    safe_reason_code IS NULL OR
                    state IN ('RESOLVING', 'FAILED', 'TIMED_OUT')
                ),
                CHECK (
                    (state = 'RUNNING' AND current_attempt_id IS NOT NULL AND
                        attempt_heartbeat_at IS NOT NULL AND first_claimed_at IS NOT NULL) OR
                    (state != 'RUNNING' AND current_attempt_id IS NULL AND
                        attempt_heartbeat_at IS NULL)
                ),
                CHECK (
                    (state IN ('READY', 'RUNNING') AND execution_spec_json IS NOT NULL) OR
                    (state NOT IN ('READY', 'RUNNING'))
                ),
                CHECK (json_valid(attempt_ids_json)),
                CHECK (json_type(attempt_ids_json) = 'array'),
                CHECK (json_array_length(attempt_ids_json) = attempt_count),
                CHECK (attempt_count <= max_attempts),
                CHECK (
                    state != 'FAILED' OR
                    (failure_category IS NOT NULL AND safe_reason_code IS NOT NULL)
                ),
                CHECK (
                    state != 'TIMED_OUT' OR
                    (timeout_kind IS NOT NULL AND safe_reason_code IS NOT NULL)
                )
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX executions_ready_profile_idx
            ON executions (resource_profile, dispatch_after, ready_at, created_at)
            WHERE state = 'READY' AND cancellation_requested = 0
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX executions_current_attempt_idx
            ON executions (current_attempt_id)
            WHERE current_attempt_id IS NOT NULL
            """
        )
        connection.execute(
            """
            CREATE INDEX executions_state_updated_idx
            ON executions (state, updated_at)
            """
        )

    def find_by_submission(
        self, request: AdmissionRequest, fingerprint: str
    ) -> Execution | None:
        """Return an equivalent prior submission, rejecting conflicting reuse.

        Args:
            request: Prototype installation/job/app identity and replay key to admit.
            fingerprint: Canonical request digest used to distinguish a replay from
                conflicting contents.

        Returns:
            Execution | None: Historical prototype execution snapshot with its pinned
                app contract and lifecycle. None indicates an absent, unsupported, or
                unmatched value as described above.
        """
        with self._connection() as connection:
            by_key = connection.execute(
                "SELECT * FROM executions WHERE idempotency_key = ?",
                (request.idempotency_key,),
            ).fetchone()
            by_identity = connection.execute(
                """
                SELECT * FROM executions
                WHERE installation_alias = ? AND empaia_job_id = ?
                """,
                (request.installation_alias, request.empaia_job_id),
            ).fetchone()
        self._validate_duplicate(request, fingerprint, by_key, by_identity)
        selected = by_key or by_identity
        return self._to_execution(selected) if selected is not None else None

    def admit(
        self,
        request: AdmissionRequest,
        fingerprint: str,
        mapping: AppMapping,
    ) -> AdmissionResult:
        """Insert one pinned execution or replay an identical key/job submission atomically.

        Args:
            request: Prototype installation/job/app identity and replay key to admit.
            fingerprint: Canonical request digest used to distinguish a replay from
                conflicting contents.
            mapping: Deployment-approved prototype app/version contract pinned at
                admission.

        Returns:
            AdmissionResult: The admitted execution and whether this operation created
                it.

        Raises:
            ValueError: A value fails the validation described above.
        """
        if mapping.max_runtime_seconds <= 0 or mapping.max_attempts <= 0:
            raise ValueError("app mapping limits must be positive")
        execution_id = str(uuid4())
        with self._transaction() as connection:
            now = self._timestamp()
            rows = connection.execute(
                """
                SELECT * FROM executions
                WHERE idempotency_key = ?
                   OR (installation_alias = ? AND empaia_job_id = ?)
                """,
                (
                    request.idempotency_key,
                    request.installation_alias,
                    request.empaia_job_id,
                ),
            ).fetchall()
            if rows:
                by_key = next(
                    (
                        row
                        for row in rows
                        if row["idempotency_key"] == request.idempotency_key
                    ),
                    None,
                )
                by_identity = next(
                    (
                        row
                        for row in rows
                        if row["installation_alias"] == request.installation_alias
                        and row["empaia_job_id"] == request.empaia_job_id
                    ),
                    None,
                )
                self._validate_duplicate(request, fingerprint, by_key, by_identity)
                return AdmissionResult(self._to_execution(rows[0]), created=False)

            connection.execute(
                """
                INSERT INTO executions (
                    id, installation_alias, empaia_job_id, registered_app_id,
                    registered_app_version, idempotency_key, request_fingerprint,
                    model_id, model_version, workflow, resource_profile,
                    max_runtime_seconds, max_attempts, input_name, output_name,
                    model_parameter_names_json, allowed_wsi_plugin_ids_json,
                    state, created_at, updated_at, resolution_after,
                    attempt_ids_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution_id,
                    request.installation_alias,
                    request.empaia_job_id,
                    request.registered_app_id,
                    request.registered_app_version,
                    request.idempotency_key,
                    fingerprint,
                    mapping.model_id,
                    mapping.model_version,
                    mapping.workflow,
                    mapping.resource_profile,
                    mapping.max_runtime_seconds,
                    mapping.max_attempts,
                    mapping.input_name,
                    mapping.output_name,
                    json.dumps(mapping.model_parameter_names),
                    json.dumps(mapping.allowed_wsi_plugin_ids),
                    ExecutionState.ACCEPTED,
                    now,
                    now,
                    now,
                    "[]",
                ),
            )
            row = self._get_row(connection, execution_id)
        return AdmissionResult(self._to_execution(row), created=True)

    @staticmethod
    def _validate_duplicate(
        request: AdmissionRequest,
        fingerprint: str,
        by_key: sqlite3.Row | None,
        by_identity: sqlite3.Row | None,
    ) -> None:
        """Reject conflicting fingerprints or a job identity reused with a different replay key.

        Args:
            request: Prototype installation/job/app identity and replay key to admit.
            fingerprint: Canonical request digest used to distinguish a replay from
                conflicting contents.
            by_key: Existing prototype row matched by replay key, if any.
            by_identity: Existing prototype row matched by installation/job identity, if
                any.

        Returns:
            None: No return value.

        Raises:
            SubmissionConflictError: The prototype replay key or job identity conflicts
                with stored admission.
        """
        for row in (by_key, by_identity):
            if row is not None and row["request_fingerprint"] != fingerprint:
                raise SubmissionConflictError(
                    "idempotency key or EMPAIA job identity was reused with "
                    "different submission data"
                )
        if (
            by_identity is not None
            and by_identity["idempotency_key"] != request.idempotency_key
        ):
            raise SubmissionConflictError(
                "EMPAIA job identity was reused with a different idempotency key"
            )

    def get(self, execution_id: str) -> Execution:
        """Read a durable row and reconstruct its typed execution snapshot.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            Execution: Historical prototype execution snapshot with its pinned app
                contract and lifecycle.
        """
        with self._connection() as connection:
            row = self._get_row(connection, execution_id)
        return self._to_execution(row)

    def check_ready(self) -> None:
        """Verify schema compatibility and the ability to commit a write lock.

        Returns:
            None: No return value.

        Raises:
            IncompatibleSchemaError: The stored prototype schema or JSON data is
                unsupported.
        """
        with self._transaction() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version != _SCHEMA_VERSION:
                raise IncompatibleSchemaError(
                    f"database schema {version} is not supported"
                )
            connection.execute("UPDATE executions SET updated_at = updated_at WHERE 0")

    def begin_resolution(self, execution_id: str) -> Execution:
        """Move an accepted execution into resolution or return its existing resolving state.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            Execution: Historical prototype execution snapshot with its pinned app
                contract and lifecycle.
        """
        with self._transaction() as connection:
            row = self._get_row(connection, execution_id)
            state = ExecutionState(row["state"])
            if state is ExecutionState.RESOLVING:
                return self._to_execution(row)
            self._require_state(row, {ExecutionState.ACCEPTED})
            now = self._timestamp()
            connection.execute(
                "UPDATE executions SET state = ?, updated_at = ? WHERE id = ?",
                (ExecutionState.RESOLVING, now, execution_id),
            )
            updated = self._get_row(connection, execution_id)
        return self._to_execution(updated)

    def list_resolution_candidates(self, limit: int = 100) -> list[Execution]:
        """List accepted/resolving records whose resolution retry time has arrived, ordered by due and creation time.

        Args:
            limit: Maximum number of eligible records to inspect or return.

        Returns:
            list[Execution]: Selected records, identifiers, or fixture values in the
                ordering described above.
        """
        if limit < 1:
            return []
        now = self._timestamp()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM executions
                WHERE state IN (?, ?) AND resolution_after <= ?
                ORDER BY resolution_after, created_at
                LIMIT ?
                """,
                (ExecutionState.ACCEPTED, ExecutionState.RESOLVING, now, limit),
            ).fetchall()
        return [self._to_execution(row) for row in rows]

    def defer_resolution(self, execution_id: str, retry_after: datetime) -> Execution:
        """Require resolving state and store the next resolution retry time.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.
            retry_after: Minimum rejection delay or absolute prototype retry time,
                according to its declared type.

        Returns:
            Execution: Historical prototype execution snapshot with its pinned app
                contract and lifecycle.
        """
        retry_time = self._normalize_time(retry_after).isoformat()
        with self._transaction() as connection:
            row = self._get_row(connection, execution_id)
            self._require_state(row, {ExecutionState.RESOLVING})
            now = self._timestamp()
            connection.execute(
                """
                UPDATE executions
                SET resolution_after = ?, updated_at = ?
                WHERE id = ?
                """,
                (retry_time, now, execution_id),
            )
            updated = self._get_row(connection, execution_id)
        return self._to_execution(updated)

    def mark_ready(self, execution_id: str, execution_spec: ExecutionSpec) -> Execution:
        """Pin a resolved specification and ready timestamp, rejecting cancellation or a contradictory already-ready specification.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.
            execution_spec: Resolved credential-free prototype specification to pin
                before dispatch.

        Returns:
            Execution: Historical prototype execution snapshot with its pinned app
                contract and lifecycle.

        Raises:
            InvalidTransitionError: The prototype state or pinned specification rejects
                the requested transition.
        """
        encoded_spec = json.dumps(
            execution_spec.to_dict(), sort_keys=True, separators=(",", ":")
        )
        with self._transaction() as connection:
            row = self._get_row(connection, execution_id)
            state = ExecutionState(row["state"])
            if state is ExecutionState.READY:
                if row["execution_spec_json"] == encoded_spec:
                    return self._to_execution(row)
                raise InvalidTransitionError(
                    "ready execution specification is immutable"
                )
            self._require_state(row, {ExecutionState.RESOLVING})
            if bool(row["cancellation_requested"]):
                raise InvalidTransitionError(
                    "a cancelled execution cannot become ready"
                )
            if row["failure_category"] is not None:
                raise InvalidTransitionError(
                    "an execution with pending failure cannot become ready"
                )
            self._require_pinned_spec(row, execution_spec)
            now = self._timestamp()
            connection.execute(
                """
                UPDATE executions
                SET state = ?, execution_spec_json = ?, resolution_after = NULL,
                    ready_at = ?, dispatch_after = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    ExecutionState.READY,
                    encoded_spec,
                    now,
                    now,
                    now,
                    execution_id,
                ),
            )
            updated = self._get_row(connection, execution_id)
        return self._to_execution(updated)

    @staticmethod
    def _require_pinned_spec(row: sqlite3.Row, spec: ExecutionSpec) -> None:
        """Ensure resolved model/workflow/runtime/output fields and parameter names match the admitted app contract.

        Args:
            row: Database row whose state is being decoded, checked, or updated.
            spec: Immutable resolved prototype specification to compare against the
                admitted contract.

        Returns:
            None: No return value.

        Raises:
            InvalidTransitionError: The prototype state or pinned specification rejects
                the requested transition.
        """
        pinned = {
            "model_id": row["model_id"],
            "model_version": row["model_version"],
            "workflow": row["workflow"],
            "resource_profile": row["resource_profile"],
            "max_runtime_seconds": row["max_runtime_seconds"],
            "output_name": row["output_name"],
        }
        actual = {name: getattr(spec, name) for name in pinned}
        allowed_parameters = set(json.loads(row["model_parameter_names_json"]))
        if actual != pinned or set(spec.model_parameters) != allowed_parameters:
            raise InvalidTransitionError(
                "resolved execution specification contradicts its admitted contract"
            )

    def fail_resolution_infrastructure(
        self, execution_id: str, reason_code: str, reason_message: str
    ) -> Execution:
        """Store a terminal infrastructure failure for an accepted or resolving execution.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.
            reason_code: Bounded safe failure identifier stored by the prototype.
            reason_message: Bounded safe user-facing failure explanation stored by the
                prototype.

        Returns:
            Execution: Historical prototype execution snapshot with its pinned app
                contract and lifecycle.
        """
        self._validate_reason(reason_code, reason_message)
        with self._transaction() as connection:
            row = self._get_row(connection, execution_id)
            self._require_state(
                row, {ExecutionState.ACCEPTED, ExecutionState.RESOLVING}
            )
            self._mark_failed(
                connection,
                row,
                FailureCategory.INFRASTRUCTURE,
                reason_code,
                reason_message,
                self._now(),
            )
            updated = self._get_row(connection, execution_id)
        return self._to_execution(updated)

    def record_resolution_application_failure(
        self, execution_id: str, reason_code: str, reason_message: str
    ) -> Execution:
        """Persist failure intent before reporting the external side effect.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.
            reason_code: Bounded safe failure identifier stored by the prototype.
            reason_message: Bounded safe user-facing failure explanation stored by the
                prototype.

        Returns:
            Execution: Historical prototype execution snapshot with its pinned app
                contract and lifecycle.

        Raises:
            InvalidTransitionError: The prototype state or pinned specification rejects
                the requested transition.
        """
        self._validate_reason(reason_code, reason_message)
        with self._transaction() as connection:
            row = self._get_row(connection, execution_id)
            self._require_state(row, {ExecutionState.RESOLVING})
            if row["failure_category"] is not None:
                if (
                    row["failure_category"] == FailureCategory.APPLICATION
                    and row["safe_reason_code"] == reason_code
                    and row["safe_reason_message"] == reason_message
                ):
                    return self._to_execution(row)
                raise InvalidTransitionError("resolution failure intent is immutable")
            now = self._timestamp()
            connection.execute(
                """
                UPDATE executions
                SET failure_category = ?, safe_reason_code = ?,
                    safe_reason_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    FailureCategory.APPLICATION,
                    reason_code,
                    reason_message,
                    now,
                    execution_id,
                ),
            )
            updated = self._get_row(connection, execution_id)
        return self._to_execution(updated)

    def settle_resolution_application_failure(self, execution_id: str) -> Execution:
        """Make a pending reported application failure terminal after external reporting completes.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            Execution: Historical prototype execution snapshot with its pinned app
                contract and lifecycle.

        Raises:
            InvalidTransitionError: The prototype state or pinned specification rejects
                the requested transition.
        """
        with self._transaction() as connection:
            row = self._get_row(connection, execution_id)
            self._require_state(row, {ExecutionState.RESOLVING})
            if row["failure_category"] != FailureCategory.APPLICATION:
                raise InvalidTransitionError(
                    "execution has no pending application failure"
                )
            now = self._timestamp()
            connection.execute(
                """
                UPDATE executions
                SET state = ?, resolution_after = NULL, updated_at = ?
                WHERE id = ?
                """,
                (ExecutionState.FAILED, now, execution_id),
            )
            updated = self._get_row(connection, execution_id)
        return self._to_execution(updated)

    def list_ready(self, resource_profile: str, limit: int) -> list[Execution]:
        """List uncancelled due ready records for one resource profile in dispatch order.

        Args:
            resource_profile: Prototype compute profile used to select eligible ready
                executions.
            limit: Maximum number of eligible records to inspect or return.

        Returns:
            list[Execution]: Selected records, identifiers, or fixture values in the
                ordering described above.
        """
        if limit < 1:
            return []
        now = self._timestamp()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM executions
                WHERE state = ? AND cancellation_requested = 0
                  AND resource_profile = ? AND dispatch_after <= ?
                ORDER BY dispatch_after, ready_at, created_at
                LIMIT ?
                """,
                (ExecutionState.READY, resource_profile, now, limit),
            ).fetchall()
        return [self._to_execution(row) for row in rows]

    def claim(
        self,
        execution_id: str,
        attempt_id: str,
        max_initial_capacity_wait_seconds: int | None = None,
    ) -> Execution:
        """Atomically and idempotently claim one ready execution.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.
            attempt_id: Unique prototype worker identity that must own the current
                attempt.
            max_initial_capacity_wait_seconds: Optional prototype wait limit before the
                first compute claim.

        Returns:
            Execution: Historical prototype execution snapshot with its pinned app
                contract and lifecycle.

        Raises:
            AttemptOwnershipError: The prototype attempt identity no longer owns the
                execution.
            InvalidTransitionError: The prototype state or pinned specification rejects
                the requested transition.
            ValueError: A value fails the validation described above.
        """
        self._validate_capacity_limit(max_initial_capacity_wait_seconds)
        if str(UUID(attempt_id)) != attempt_id:
            raise ValueError("attempt identity must be a canonical UUID")
        with self._transaction() as connection:
            row = self._get_row(connection, execution_id)
            if (
                ExecutionState(row["state"]) is ExecutionState.RUNNING
                and row["current_attempt_id"] == attempt_id
            ):
                now = self._now()
                if self._runtime_expired(row, now):
                    self._mark_timed_out(
                        connection, row, TimeoutKind.MODEL_RUNTIME, now
                    )
                    return self._to_execution(self._get_row(connection, execution_id))
                if bool(row["cancellation_requested"]):
                    return self._to_execution(row)
                now_text = now.isoformat()
                connection.execute(
                    """
                    UPDATE executions
                    SET attempt_heartbeat_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (now_text, now_text, execution_id),
                )
                return self._to_execution(self._get_row(connection, execution_id))
            self._require_state(row, {ExecutionState.READY})
            if bool(row["cancellation_requested"]):
                raise InvalidTransitionError("cancelled execution cannot be claimed")
            used_attempt_ids = json.loads(row["attempt_ids_json"])
            if attempt_id in used_attempt_ids:
                raise AttemptOwnershipError("attempt identity has already been fenced")
            now = self._now()
            timeout_kind = self._timeout_kind(
                row, now, max_initial_capacity_wait_seconds
            )
            if timeout_kind is not None:
                self._mark_timed_out(connection, row, timeout_kind, now)
                return self._to_execution(self._get_row(connection, execution_id))
            if int(row["attempt_count"]) >= int(row["max_attempts"]):
                self._mark_failed(
                    connection,
                    row,
                    FailureCategory.INFRASTRUCTURE,
                    "attempts_exhausted",
                    "execution attempt limit was exhausted",
                    now,
                )
                return self._to_execution(self._get_row(connection, execution_id))
            now_text = now.isoformat()
            try:
                connection.execute(
                    """
                    UPDATE executions
                    SET state = ?, current_attempt_id = ?,
                        attempt_heartbeat_at = ?, attempt_count = attempt_count + 1,
                        attempt_ids_json = ?,
                        first_claimed_at = COALESCE(first_claimed_at, ?),
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        ExecutionState.RUNNING,
                        attempt_id,
                        now_text,
                        json.dumps([*used_attempt_ids, attempt_id]),
                        now_text,
                        now_text,
                        execution_id,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise AttemptOwnershipError(
                    "attempt identity is already in use"
                ) from error
            updated = self._get_row(connection, execution_id)
        return self._to_execution(updated)

    def heartbeat_attempt(self, execution_id: str, attempt_id: str) -> Execution:
        """Validate current ownership and update its heartbeat, or settle cancellation/runtime expiry.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.
            attempt_id: Unique prototype worker identity that must own the current
                attempt.

        Returns:
            Execution: Historical prototype execution snapshot with its pinned app
                contract and lifecycle.
        """
        with self._transaction() as connection:
            row = self._get_row(connection, execution_id)
            self._require_attempt(row, attempt_id)
            now = self._now()
            if bool(row["cancellation_requested"]):
                self._mark_cancelled(connection, row, now)
            elif self._runtime_expired(row, now):
                self._mark_timed_out(connection, row, TimeoutKind.MODEL_RUNTIME, now)
            else:
                now_text = now.isoformat()
                connection.execute(
                    """
                    UPDATE executions
                    SET attempt_heartbeat_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (now_text, now_text, execution_id),
                )
            updated = self._get_row(connection, execution_id)
        return self._to_execution(updated)

    def list_stale_attempts(
        self, stale_before: datetime, limit: int = 100
    ) -> list[Execution]:
        """List running executions with heartbeats no newer than the supplied cutoff.

        Args:
            stale_before: Latest heartbeat timestamp that qualifies a prototype attempt
                for recovery.
            limit: Maximum number of eligible records to inspect or return.

        Returns:
            list[Execution]: Selected records, identifiers, or fixture values in the
                ordering described above.
        """
        cutoff = self._normalize_time(stale_before).isoformat()
        if limit < 1:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM executions
                WHERE state = ? AND attempt_heartbeat_at <= ?
                ORDER BY attempt_heartbeat_at
                LIMIT ?
                """,
                (ExecutionState.RUNNING, cutoff, limit),
            ).fetchall()
        return [self._to_execution(row) for row in rows]

    def recover_attempt(
        self,
        execution_id: str,
        attempt_id: str,
        stale_before: datetime,
        retry_after: datetime,
    ) -> Execution:
        """Fence a lost worker and retry without resetting its runtime clock.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.
            attempt_id: Unique prototype worker identity that must own the current
                attempt.
            stale_before: Latest heartbeat timestamp that qualifies a prototype attempt
                for recovery.
            retry_after: Minimum rejection delay or absolute prototype retry time,
                according to its declared type.

        Returns:
            Execution: Historical prototype execution snapshot with its pinned app
                contract and lifecycle.

        Raises:
            AttemptOwnershipError: The prototype attempt identity no longer owns the
                execution.
        """
        stale_cutoff = self._normalize_time(stale_before)
        retry_time = self._normalize_time(retry_after)
        with self._transaction() as connection:
            row = self._get_row(connection, execution_id)
            self._require_attempt(row, attempt_id)
            heartbeat = self._parse_timestamp(row["attempt_heartbeat_at"])
            if heartbeat is None or heartbeat > stale_cutoff:
                raise AttemptOwnershipError("attempt is no longer stale")
            now = self._now()
            if bool(row["cancellation_requested"]):
                self._mark_cancelled(connection, row, now)
            elif self._runtime_expired(row, now):
                self._mark_timed_out(connection, row, TimeoutKind.MODEL_RUNTIME, now)
            elif int(row["attempt_count"]) >= int(row["max_attempts"]):
                self._mark_failed(
                    connection,
                    row,
                    FailureCategory.INFRASTRUCTURE,
                    "attempts_exhausted",
                    "execution attempt limit was exhausted",
                    now,
                )
            else:
                dispatch_after = max(now, retry_time).isoformat()
                connection.execute(
                    """
                    UPDATE executions
                    SET state = ?, current_attempt_id = NULL,
                        attempt_heartbeat_at = NULL, dispatch_after = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        ExecutionState.READY,
                        dispatch_after,
                        now.isoformat(),
                        execution_id,
                    ),
                )
            updated = self._get_row(connection, execution_id)
        return self._to_execution(updated)

    def acknowledge_cancellation(self, execution_id: str, attempt_id: str) -> Execution:
        """Require the current attempt and a cancellation request before terminal cancellation.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.
            attempt_id: Unique prototype worker identity that must own the current
                attempt.

        Returns:
            Execution: Historical prototype execution snapshot with its pinned app
                contract and lifecycle.

        Raises:
            InvalidTransitionError: The prototype state or pinned specification rejects
                the requested transition.
        """
        with self._transaction() as connection:
            row = self._get_row(connection, execution_id)
            self._require_attempt(row, attempt_id)
            if not bool(row["cancellation_requested"]):
                raise InvalidTransitionError("execution has no cancellation request")
            self._mark_cancelled(connection, row, self._now())
            updated = self._get_row(connection, execution_id)
        return self._to_execution(updated)

    def request_cancellation(self, execution_id: str) -> Execution:
        """Record cancellation intent or immediately cancel eligible early states, preserving terminal outcomes.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            Execution: Historical prototype execution snapshot with its pinned app
                contract and lifecycle.
        """
        with self._transaction() as connection:
            row = self._get_row(connection, execution_id)
            state = ExecutionState(row["state"])
            if state.terminal:
                return self._to_execution(row)
            now = self._now()
            if (
                state is ExecutionState.RESOLVING
                and row["failure_category"] == FailureCategory.APPLICATION
            ):
                connection.execute(
                    """
                    UPDATE executions
                    SET cancellation_requested = 1, updated_at = ?
                    WHERE id = ?
                    """,
                    (now.isoformat(), execution_id),
                )
            elif state in _EARLY_CANCELLATION_STATES:
                self._mark_cancelled(connection, row, now)
            else:
                connection.execute(
                    """
                    UPDATE executions
                    SET cancellation_requested = 1, updated_at = ?
                    WHERE id = ?
                    """,
                    (now.isoformat(), execution_id),
                )
            updated = self._get_row(connection, execution_id)
        return self._to_execution(updated)

    def settle_requested_cancellations(self) -> list[Execution]:
        """Fence running attempts after cancellation, independent of heartbeats.

        Returns:
            list[Execution]: Selected records, identifiers, or fixture values in the
                ordering described above.
        """
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT * FROM executions
                WHERE state = ? AND cancellation_requested = 1
                """,
                (ExecutionState.RUNNING,),
            ).fetchall()
            now = self._now()
            for row in rows:
                self._mark_cancelled(connection, row, now)
            return [
                self._to_execution(self._get_row(connection, row["id"])) for row in rows
            ]

    def expire_timeouts(
        self, max_initial_capacity_wait_seconds: int | None
    ) -> list[Execution]:
        """Apply current capacity policy and pinned model runtime limits.

        Args:
            max_initial_capacity_wait_seconds: Optional prototype wait limit before the
                first compute claim.

        Returns:
            list[Execution]: Selected records, identifiers, or fixture values in the
                ordering described above.
        """
        self._validate_capacity_limit(max_initial_capacity_wait_seconds)
        expired_ids: list[str] = []
        with self._transaction() as connection:
            now = self._now()
            rows = connection.execute(
                """
                SELECT * FROM executions
                WHERE state IN (?, ?)
                """,
                (ExecutionState.READY, ExecutionState.RUNNING),
            ).fetchall()
            for row in rows:
                if bool(row["cancellation_requested"]):
                    self._mark_cancelled(connection, row, now)
                    continue
                kind = self._timeout_kind(row, now, max_initial_capacity_wait_seconds)
                if kind is None:
                    continue
                self._mark_timed_out(connection, row, kind, now)
                expired_ids.append(row["id"])
            expired = [
                self._to_execution(self._get_row(connection, execution_id))
                for execution_id in expired_ids
            ]
        return expired

    @staticmethod
    def _mark_failed(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        category: FailureCategory,
        reason_code: str,
        reason_message: str,
        now: datetime,
    ) -> None:
        """Store a bounded safe failure reason and release worker ownership in the caller transaction.

        Args:
            connection: Database connection owned by the enclosing operation or
                transaction.
            row: Database row whose state is being decoded, checked, or updated.
            category: Application or infrastructure classification for the stored
                failure.
            reason_code: Bounded safe failure identifier stored by the prototype.
            reason_message: Bounded safe user-facing failure explanation stored by the
                prototype.
            now: Aware current time supplied by the repository or validation clock.

        Returns:
            None: No return value.
        """
        SQLiteExecutionStore._validate_reason(reason_code, reason_message)
        connection.execute(
            """
            UPDATE executions
            SET state = ?, current_attempt_id = NULL,
                attempt_heartbeat_at = NULL, failure_category = ?,
                timeout_kind = NULL, safe_reason_code = ?,
                safe_reason_message = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                ExecutionState.FAILED,
                category,
                reason_code,
                reason_message,
                now.isoformat(),
                row["id"],
            ),
        )

    @staticmethod
    def _mark_timed_out(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        kind: TimeoutKind,
        now: datetime,
    ) -> None:
        """Store a capacity/runtime timeout reason and release worker ownership.

        Args:
            connection: Database connection owned by the enclosing operation or
                transaction.
            row: Database row whose state is being decoded, checked, or updated.
            kind: Input access adapter name or prototype timeout category.
            now: Aware current time supplied by the repository or validation clock.

        Returns:
            None: No return value.
        """
        code = (
            "initial_capacity_timeout"
            if kind is TimeoutKind.INITIAL_CAPACITY
            else "model_runtime_timeout"
        )
        message = (
            "initial compute capacity wait limit was exceeded"
            if kind is TimeoutKind.INITIAL_CAPACITY
            else "model runtime limit was exceeded"
        )
        connection.execute(
            """
            UPDATE executions
            SET state = ?, current_attempt_id = NULL,
                attempt_heartbeat_at = NULL, failure_category = NULL,
                timeout_kind = ?,
                safe_reason_code = ?, safe_reason_message = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                ExecutionState.TIMED_OUT,
                kind,
                code,
                message,
                now.isoformat(),
                row["id"],
            ),
        )

    @staticmethod
    def _mark_cancelled(
        connection: sqlite3.Connection, row: sqlite3.Row, now: datetime
    ) -> None:
        """Store terminal cancellation and clear ownership and previous error metadata.

        Args:
            connection: Database connection owned by the enclosing operation or
                transaction.
            row: Database row whose state is being decoded, checked, or updated.
            now: Aware current time supplied by the repository or validation clock.

        Returns:
            None: No return value.
        """
        connection.execute(
            """
            UPDATE executions
            SET state = ?, cancellation_requested = 1,
                current_attempt_id = NULL, attempt_heartbeat_at = NULL,
                failure_category = NULL, timeout_kind = NULL,
                safe_reason_code = NULL, safe_reason_message = NULL,
                updated_at = ?
            WHERE id = ?
            """,
            (ExecutionState.CANCELLED, now.isoformat(), row["id"]),
        )

    @staticmethod
    def _timeout_kind(
        row: sqlite3.Row,
        now: datetime,
        max_initial_capacity_wait_seconds: int | None,
    ) -> TimeoutKind | None:
        """Check runtime since the first claim, or optional capacity wait since ready if never claimed.

        Args:
            row: Database row whose state is being decoded, checked, or updated.
            now: Aware current time supplied by the repository or validation clock.
            max_initial_capacity_wait_seconds: Optional prototype wait limit before the
                first compute claim.

        Returns:
            TimeoutKind | None: The applicable prototype capacity or runtime timeout
                category. None indicates an absent, unsupported, or unmatched value as
                described above.
        """
        first_claimed_at = SQLiteExecutionStore._parse_timestamp(
            row["first_claimed_at"]
        )
        if first_claimed_at is not None:
            if now - first_claimed_at >= timedelta(
                seconds=int(row["max_runtime_seconds"])
            ):
                return TimeoutKind.MODEL_RUNTIME
            return None
        if max_initial_capacity_wait_seconds is None:
            return None
        ready_at = SQLiteExecutionStore._parse_timestamp(row["ready_at"])
        if ready_at is not None and now - ready_at >= timedelta(
            seconds=max_initial_capacity_wait_seconds
        ):
            return TimeoutKind.INITIAL_CAPACITY
        return None

    @staticmethod
    def _runtime_expired(row: sqlite3.Row, now: datetime) -> bool:
        """Check only the pinned model runtime deadline, without a capacity-wait limit.

        Args:
            row: Database row whose state is being decoded, checked, or updated.
            now: Aware current time supplied by the repository or validation clock.

        Returns:
            bool: True when the prototype's pinned runtime has expired since its first
                claim.
        """
        return (
            SQLiteExecutionStore._timeout_kind(row, now, None)
            is TimeoutKind.MODEL_RUNTIME
        )

    @staticmethod
    def _validate_capacity_limit(value: int | None) -> None:
        """Require a positive initial-capacity wait limit or no limit.

        Args:
            value: Value being validated, serialized, converted, or checked by this
                helper.

        Returns:
            None: No return value.

        Raises:
            ValueError: A value fails the validation described above.
        """
        if value is not None and value <= 0:
            raise ValueError("initial capacity wait must be positive or absent")

    @staticmethod
    def _validate_reason(reason_code: str, reason_message: str) -> None:
        """Require nonempty safe reason fields within their stored size bounds.

        Args:
            reason_code: Bounded safe failure identifier stored by the prototype.
            reason_message: Bounded safe user-facing failure explanation stored by the
                prototype.

        Returns:
            None: No return value.

        Raises:
            ValueError: A value fails the validation described above.
        """
        if (
            not reason_code
            or not reason_message
            or len(reason_code) > 128
            or len(reason_message) > 512
        ):
            raise ValueError("safe failure reason is missing or too long")

    @staticmethod
    def _get_row(connection: sqlite3.Connection, execution_id: str) -> sqlite3.Row:
        """Read a row by execution ID or raise ExecutionNotFoundError.

        Args:
            connection: Database connection owned by the enclosing operation or
                transaction.
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            sqlite3.Row: Stored prototype execution row.

        Raises:
            ExecutionNotFoundError: The prototype execution ID does not exist.
        """
        row = connection.execute(
            "SELECT * FROM executions WHERE id = ?", (execution_id,)
        ).fetchone()
        if row is None:
            raise ExecutionNotFoundError(f"execution {execution_id} was not found")
        return row

    @staticmethod
    def _require_state(row: sqlite3.Row, expected: set[ExecutionState]) -> None:
        """Reject a row whose current lifecycle state is outside the permitted set.

        Args:
            row: Database row whose state is being decoded, checked, or updated.
            expected: Required status, work kind, or permitted lifecycle states for this
                check.

        Returns:
            None: No return value.

        Raises:
            InvalidTransitionError: The prototype state or pinned specification rejects
                the requested transition.
        """
        actual = ExecutionState(row["state"])
        if actual not in expected:
            names = ", ".join(sorted(state.value for state in expected))
            raise InvalidTransitionError(
                f"execution in {actual.value} must be in one of: {names}"
            )

    @staticmethod
    def _require_attempt(row: sqlite3.Row, attempt_id: str) -> None:
        """Require running state and matching current attempt identity.

        Args:
            row: Database row whose state is being decoded, checked, or updated.
            attempt_id: Unique prototype worker identity that must own the current
                attempt.

        Returns:
            None: No return value.

        Raises:
            AttemptOwnershipError: The prototype attempt identity no longer owns the
                execution.
        """
        if (
            ExecutionState(row["state"]) is not ExecutionState.RUNNING
            or row["current_attempt_id"] != attempt_id
        ):
            raise AttemptOwnershipError("attempt no longer owns this execution")

    def _now(self) -> datetime:
        """Read the injected clock and normalize its timezone to UTC.

        Returns:
            datetime: Injected current time normalized to UTC.
        """
        return self._normalize_time(self._clock())

    def _timestamp(self) -> str:
        """Render the normalized current time as ISO 8601 for SQLite storage.

        Returns:
            str: Current UTC timestamp rendered in ISO 8601 for SQLite storage.
        """
        return self._now().isoformat()

    @staticmethod
    def _normalize_time(value: datetime) -> datetime:
        """Require an aware datetime and convert it to UTC.

        Args:
            value: Value being validated, serialized, converted, or checked by this
                helper.

        Returns:
            datetime: The supplied aware datetime converted to UTC.

        Raises:
            ValueError: A value fails the validation described above.
        """
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("execution store times must be timezone-aware")
        return value.astimezone(UTC)

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        """Decode an optional stored ISO timestamp.

        Args:
            value: Value being validated, serialized, converted, or checked by this
                helper.

        Returns:
            datetime | None: Parsed stored datetime, or None when the column is absent.
        """
        return datetime.fromisoformat(value) if value is not None else None

    @staticmethod
    def _decode_json(value: str | None) -> dict[str, Any] | None:
        """Decode an optional stored JSON object, rejecting other shapes as incompatible data.

        Args:
            value: Value being validated, serialized, converted, or checked by this
                helper.

        Returns:
            dict[str, Any] | None: Decoded, serialized, or recorded fields described
                above. None indicates an absent, unsupported, or unmatched value as
                described above.

        Raises:
            IncompatibleSchemaError: The stored prototype schema or JSON data is
                unsupported.
        """
        parsed: Any = json.loads(value) if value is not None else None
        if parsed is not None and not isinstance(parsed, dict):
            raise IncompatibleSchemaError("durable JSON value is not an object")
        return parsed

    @staticmethod
    def _to_execution(row: sqlite3.Row) -> Execution:
        """Reconstruct the pinned contract, typed lifecycle values, timestamps, and optional worker specification from a row.

        Args:
            row: Database row whose state is being decoded, checked, or updated.

        Returns:
            Execution: Historical prototype execution snapshot with its pinned app
                contract and lifecycle.
        """
        raw_spec = SQLiteExecutionStore._decode_json(row["execution_spec_json"])
        return Execution(
            id=row["id"],
            installation_alias=row["installation_alias"],
            empaia_job_id=row["empaia_job_id"],
            registered_app_id=row["registered_app_id"],
            registered_app_version=row["registered_app_version"],
            idempotency_key=row["idempotency_key"],
            request_fingerprint=row["request_fingerprint"],
            model_id=row["model_id"],
            model_version=row["model_version"],
            workflow=row["workflow"],
            resource_profile=row["resource_profile"],
            max_runtime_seconds=row["max_runtime_seconds"],
            max_attempts=row["max_attempts"],
            input_name=row["input_name"],
            output_name=row["output_name"],
            model_parameter_names=tuple(json.loads(row["model_parameter_names_json"])),
            allowed_wsi_plugin_ids=tuple(
                json.loads(row["allowed_wsi_plugin_ids_json"])
            ),
            state=ExecutionState(row["state"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            resolution_after=SQLiteExecutionStore._parse_timestamp(
                row["resolution_after"]
            ),
            ready_at=SQLiteExecutionStore._parse_timestamp(row["ready_at"]),
            first_claimed_at=SQLiteExecutionStore._parse_timestamp(
                row["first_claimed_at"]
            ),
            current_attempt_id=row["current_attempt_id"],
            attempt_heartbeat_at=SQLiteExecutionStore._parse_timestamp(
                row["attempt_heartbeat_at"]
            ),
            dispatch_after=SQLiteExecutionStore._parse_timestamp(row["dispatch_after"]),
            attempt_count=row["attempt_count"],
            cancellation_requested=bool(row["cancellation_requested"]),
            failure_category=(
                FailureCategory(row["failure_category"])
                if row["failure_category"] is not None
                else None
            ),
            timeout_kind=(
                TimeoutKind(row["timeout_kind"])
                if row["timeout_kind"] is not None
                else None
            ),
            safe_reason_code=row["safe_reason_code"],
            safe_reason_message=row["safe_reason_message"],
            execution_spec=(
                ExecutionSpec.from_dict(raw_spec) if raw_spec is not None else None
            ),
        )
