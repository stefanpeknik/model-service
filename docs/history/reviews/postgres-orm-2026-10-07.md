# PostgreSQL ORM refactor, 2026-10-07

## Scope

The user requested SQLAlchemy ORM instead of handwritten persistence SQL.
The repository interface, domain records, execution transitions, HTTP contracts,
retry defaults, and workflow-only scheduling scope are unchanged. `models/`,
sibling repositories, and the deployment were not modified.

- `execution_scheduler/persistence/models.py` maps the execution table with typed
  enum attributes, UUIDs, timestamps, encrypted context bytes, and validated
  invocation/result JSONB. Enum values and JSON encodings match existing records.
  New tables use non-native SQLAlchemy enums with unbounded string columns;
  existing text columns require no conversion.
- `persistence/postgres.py` queries mapped rows and persists state through ORM
  attributes. Each operation owns one session and database transaction. NullPool
  preserves the previous connection-per-transaction lifecycle, with no shared
  session or idle connections requiring application shutdown changes. Psycopg
  remains the database driver and accepts the existing URL and libpq DSN formats.
- Database-time deadlines, `SKIP LOCKED` row selection, transaction-scoped
  advisory locks, atomic idempotent admission, and fenced updates retain their
  previous semantics. State decisions remain in `transitions.py`.
- `persistence/schema.py` creates tables from mapped metadata and retains the
  two existing additive upgrades through Alembic operations. Initialization is
  serialized per schema. This does not introduce a versioned migration system,
  rewrite existing rows, or automatically reconcile arbitrary schema changes.
- The scheduler image recipe and project dependencies include SQLAlchemy's
  asyncio support and Alembic. Existing locked package versions were retained.

## Local verification

Tests used an isolated PostgreSQL 16 container bound to loopback, with a fresh
random schema for each repository test. No deployed database, MLflow, AppService,
Ray service, Git host, or registry was contacted by these tests. Dependency
installation read the Python package index, using the repository's `.cache/uv`
and `.venv`.

- All 31 PostgreSQL repository tests passed, including the shared repository
  scenarios and nine added ORM regression tests. Coverage includes concurrent
  admission and claims, a processing limit greater than one, nonblocking row
  locks, transaction rollback and lock release, leases, fencing, deadlines,
  cancellation, encrypted context, isolated schemas, and database constraints.
- A pre-ORM SQL fixture verifies that initialization leaves the stored row
  unchanged, that invocation/result/context values load correctly, that an
  equivalent admission replays it, and that pending provider finish completes
  without rerunning processing. Another scenario removes the two historically
  optional columns and verifies concurrent, repeatable initialization restores
  them and the indexes without losing the record.
- Full top-level discovery ran 189 tests: 184 passed and 5 skipped. Four skips
  require the native heatmap opt-in; one requires libvips. Ray and deployed
  integration opt-ins were disabled. This run also covers the October 7 source
  moves that were previously checked only statically.
- Both assembled durable HTTP integration tests passed against the same local
  database with in-process provider and workflow HTTP fakes. Their gateway fake
  initially lacked the existing `set_before_mutation` contract; the fixture now
  invokes that guard before publication/finalization. No EMPAIA runtime behavior
  was changed, and these tests do not establish real EMPAIA interoperability.
- Ruff lint and formatting passed for the changed Python files. Mypy passed
  for the three ORM persistence files with imported modules followed silently.
  `git diff --check` passed.

The passing commands were:

```sh
UV_CACHE_DIR=.cache/uv uv sync --locked
# TEST_POSTGRES_DSN must identify an isolated local test database.
UV_CACHE_DIR=.cache/uv uv run --locked --offline python -m unittest \
  tests.test_postgres_execution_repository -v
UV_CACHE_DIR=.cache/uv uv run --locked --offline python -m unittest \
  discover -s tests -v
UV_CACHE_DIR=.cache/uv uv run --locked --offline python -m unittest \
  tests.integration.test_durable_empaia_execution -v
UV_CACHE_DIR=.cache/uv uv run --locked --offline ruff check --no-fix \
  execution_scheduler/persistence tests/test_postgres_execution_repository.py \
  tests/integration/test_durable_empaia_execution.py
UV_CACHE_DIR=.cache/uv uv run --locked --offline ruff format --check \
  execution_scheduler/persistence tests/test_postgres_execution_repository.py \
  tests/integration/test_durable_empaia_execution.py
UV_CACHE_DIR=.cache/uv uv run --locked --offline mypy --follow-imports=silent \
  execution_scheduler/persistence/models.py \
  execution_scheduler/persistence/postgres.py \
  execution_scheduler/persistence/schema.py
```

## Deployment boundary

No images were built or pushed, no Helm release changed, and no Git changes
were pushed. The October 6 deployed evidence still describes its original
runtime, not this ORM adapter. Rebuild the scheduler image with its new
dependencies before deploying this source. The source reorganization also
requires the Ray image rebuilds described in `docs/README.md`.

EMPAIA remains the next phase. Its outstanding integration decisions and
acceptance criteria are unchanged.
