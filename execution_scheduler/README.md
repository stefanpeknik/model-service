# Execution scheduler package

This package owns durable scheduled executions. Callers use `schedule`, `get`,
and `cancel`; a background worker claims due work and advances it through
workflow processing and provider finish.

## Start here

- `main.py` exposes the ASGI application used by the container.
- `runtime.py` wires production adapters and manages worker lifetime.
- `api.py` translates HTTP requests and failures to the scheduler interface.
- `scheduler.py` implements admission, status reads, cancellation, and public
  execution views.
- `worker.py` owns lease renewal, dispatch, settlement, and local cancellation.
- `active_work.py` tracks locally running attempts so cancellation can stop
  their tasks without making in-memory state authoritative.

## Durable state

- `contracts.py` contains caller-visible request, result, and status types.
- `domain.py` contains internal records, claims, retry policies, and settlement
  values.
- `transitions.py` is the only module that creates or changes execution state,
  including admission, claims, cancellation, settlement, lease recovery, and
  deadlines.
- `repository.py` defines the seven persistence operations and provides the
  in-memory test adapter.
- `postgres.py` implements those operations with PostgreSQL.
- `postgres_schema.py` owns PostgreSQL schema creation.
- `security.py` encrypts protected provider context at rest.

## External adapters

- `dispatch.py` invokes workflow applications through Ray Serve HTTP.
- `providers.py` defines the provider-adapter interface and finish outcomes.
- `standalone.py` prepares shared-storage schedule requests.
- `../integrations/empaia/provider.py` prepares and finishes EMPAIA-backed
  executions.
- `../integrations/empaia/http_gateway.py` implements the AppService HTTP and
  direct-access boundary.
- `../integrations/empaia/image_io.py` materializes rectangle masks and
  publishes TIFF artifacts as EMPAIA pixelmaps.
- `health.py` checks declared workflow applications against Ray Serve routes.
- `synchronous.py` runs the same workflow contract without creating a durable
  execution.

The scheduler and worker depend on repository, dispatch, and provider
interfaces. Concrete adapters translate their library-specific failures before
those failures reach orchestration. Keep state decisions in `transitions.py`;
PostgreSQL and the in-memory adapter persist the resulting record.
