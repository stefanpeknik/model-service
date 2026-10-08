# Public application package

This package owns the public HTTP application and wires its dependencies.
Durable execution policy remains in `../execution_scheduler/`; workflow
processing remains in `../workflow_execution/` and `../workflows/`.

## Start here

- `main.py` exposes the ASGI application at `model_service.main:app`.
- `runtime.py` constructs the catalog, repository, providers, workflow dispatch,
  inference proxy, and health checker. Its lifespan initializes persistence,
  owns the scheduler's running scope, and closes the shared HTTP client.
- `settings.py` uses Pydantic Settings for environment configuration and field
  validation. Existing setting names, numeric-seconds units, and defaults remain
  unchanged. `from_environment()` reads the process environment; an explicit
  mapping replaces it completely, including an empty mapping. Direct construction
  validates Python field values without reading environment variables. Counts
  and durations obey their declared bounds; the poll interval must be finite.
- `api.py` translates HTTP requests and failures. It exposes immediate workflow
  invocation, durable admission, EMPAIA admission, polling, cancellation, and
  health routes. It includes the legacy inference router after specific routes.
- `inference.py` forwards existing model and slide requests without creating
  an execution record.
- `synchronous.py` validates and dispatches an immediate workflow invocation
  without admitting durable work.
- `health.py` reports downstream workflow reachability separately from durable
  API readiness.

Both immediate invocation and scheduled processing use
`workflow_execution.dispatch.WorkflowDispatch`. Provider preparation and finish
belong to `../integrations/`, not HTTP handlers or workflow processing.

The package move changes Python import paths, not public HTTP routes, settings,
execution states, or database schema. The deployment Dockerfile now starts
`model_service.main:app`; rebuild it before using the new source layout.
