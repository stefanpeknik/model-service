# Execution scheduler package

This package owns durable scheduled executions. Callers use `schedule`, `get`,
and `cancel`; a background worker claims due work and advances it through
workflow processing and provider finish.

## Start here

- The public HTTP application and dependency wiring are in
  [`../model_service/`](../model_service/README.md). This package owns durable
  executions, not raw inference or immediate workflow execution.
- `scheduler.py` implements admission, status reads, cancellation, and public
  execution views. Its `running()` context owns worker lifetime and shares a
  private cancellation registry with that worker. Construction starts no tasks.
- `worker.py` owns lease renewal, dispatch, settlement, and local cancellation.
  Maintenance runs independently of its single active claim, so queued
  deadlines and abandoned leases advance while processing is blocked.
- `active_work.py` tracks locally running attempts so cancellation can stop
  their tasks without making in-memory state authoritative. Callers do not wire
  this registry themselves.

## Durable state

- `contracts.py` contains caller-visible request, result, and status types.
- `domain.py` contains internal records, claims, retry policies, and settlement
  values.
- `transitions.py` is the only module that creates or changes execution state,
  including admission, claims, cancellation, settlement, lease recovery, and
  deadlines.
- `repository.py` defines persistence and maintenance operations and their
  availability error. It imports neither persistence implementation.
- `persistence/memory.py` implements the interface in memory.
- `persistence/postgres.py` implements the interface with SQLAlchemy ORM sessions
  over PostgreSQL. Each operation has its own transaction and connection; no
  session or idle pooled connection is shared between worker tasks.
- `persistence/models.py` defines the typed execution row, database constraints
  and indexes, and JSONB conversion for workflow invocations and results. ORM
  rows stay inside persistence; callers receive the domain records above.
- `persistence/schema.py` creates the mapped schema and indexes. It uses Alembic
  operations for the two pre-existing additive column upgrades, under the same
  initialization advisory lock. It does not rewrite stored records or maintain
  an Alembic revision history.
- `persistence/protected_context.py` encrypts protected provider context at rest.

## External adapters

- `../workflow_execution/dispatch.py` invokes workflow applications through Ray
  Serve HTTP. Both immediate and scheduled execution use this interface.
- `providers.py` defines the provider-adapter interface and finish outcomes.
- `../integrations/standalone.py` prepares shared-storage schedule requests.
- `../integrations/empaia/provider.py` prepares and finishes EMPAIA-backed
  executions.
- `../integrations/empaia/http_gateway.py` implements the AppService HTTP and
  direct-access boundary.
- `../integrations/empaia/responses.py` validates consumed AppService response
  fields; `materialization.py` owns local input linking and writing.
- `../integrations/empaia/image_io.py` materializes rectangle masks and
  publishes TIFF artifacts as EMPAIA pixelmaps.
- `../model_service/health.py` reports Ray availability and missing workflow
  routes through `/healthz/workflows`. Durable API readiness at `/readyz` checks
  repository access without requiring Ray.

The scheduler and worker depend on repository, dispatch, and provider
interfaces. Concrete adapters translate their library-specific failures before
those failures reach orchestration. Keep state decisions in `transitions.py`;
PostgreSQL and the in-memory adapter persist the resulting record.

For sequential claims, due provider finish and terminal reports precede
processing. Processing workflows are considered by their oldest eligible
record's creation time; within that workflow, due time precedes creation time.
Provider work is also ordered by due time, then creation time. Processing at
capacity does not block another workflow. PostgreSQL locking can skip busy
records, so concurrent claims do not promise global FIFO. Both adapters run the
shared scenarios in `tests/repository_contract.py`; PostgreSQL concurrency
checks remain in `tests/test_postgres_execution_repository.py`.
The [October 7 ORM regression record](../docs/reviews/postgres-orm-2026-10-07.md)
covers local concurrency, schema compatibility, and application checks. It is
not a deployment validation of the ORM adapter.

Leaving `scheduler.running()` stops local work and joins worker tasks. It does
not cancel durable executions; another worker can recover their expired leases.
The context cannot be entered twice concurrently, but can be restarted after
shutdown. Repository, dispatch, and provider resources remain caller-owned.
