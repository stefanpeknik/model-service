# Execution database migrations

Yoyo owns database creation and upgrades. `../models.py` describes the current
ORM mapping; changing a mapped column alone does not change an existing database.
Application queries remain ORM-based. SQL here describes schema changes only.

## Apply migrations

Scheduler startup applies pending migrations automatically before starting its
worker. To apply them separately, set `DATABASE_URL` through your normal secret
configuration, then run from the repository root:

```sh
UV_CACHE_DIR=.cache/uv uv run --locked --offline python \
  -m model_service.scheduler.store.schema
```

The default schema is `model_service`. `--schema NAME` selects an isolated target
for local tests. PostgreSQL URLs and libpq DSNs are both supported. Credentials
are read from the environment, not passed as command-line arguments.

The initializer creates the target schema and serializes startup with a
PostgreSQL advisory lock. Yoyo also locks migration execution and tracks applied
versions in that schema, independently of other test or application schemas.
Each migration is transactional; the entire sequence is not one transaction.
An initialization failure prevents scheduler startup.

## Change the schema

1. Update the ORM mapping in `../models.py`.
2. Add the next numbered migration here, for example
   `0002-add-example-column.sql`. Declare its predecessor and write the upgrade:

   ```sql
   -- Add an optional example column.
   -- depends: 0001-create-executions

   ALTER TABLE executions ADD COLUMN example TEXT;
   ```

   Tables are unqualified because the initializer sets the target search path.
   A template can also be created with `uv run yoyo new --sql` and this directory
   as its destination. No `yoyo.ini` or credentials file is required.
3. Test both a fresh database and a database at the previous revision. Preserve
   records and defaults, check repeated startup, and check failure rollback.
   `tests/test_postgres_execution_repository.py` has examples. Its
   `TEST_POSTGRES_DSN` must point to a disposable local test database.

   ```sh
   UV_CACHE_DIR=.cache/uv uv run --locked --offline python -m unittest \
     tests.scheduler.test_postgres_execution_repository -v
   ```

Never edit or rename an applied migration. Add a new migration instead. SQL
rollback files are optional; this queue does not perform automatic downgrade or
data-deleting rollback. Coordinate incompatible changes with application
deployment rather than running old and new schedulers against incompatible DDL.

## WIP baseline and interrupted migrations

`0001-create-executions.sql` is the frozen baseline for this WIP codebase. The
old metadata-based initializer and its historical missing-column repairs are
removed. An existing execution table without Yoyo history makes initialization
fail; nothing automatically deletes records or marks the baseline as applied.

Before deploying this change, choose fresh disposable storage, explicitly reset
storage whose data can be discarded, or verify the retained schema against the
baseline before a controlled Yoyo adoption. Do not mark an arbitrary schema as
current. No deployed database was reset as part of this change.

If a migration process is killed, Yoyo can retain its lock row. Confirm that no
migration process is still active before using Yoyo's `break-lock` command.
Do not automatically break locks or mark failed migrations as applied. After an
interruption, inspect both schema and history before retrying; migration DDL and
its history marker are separate commits in Yoyo.

See the [Yoyo documentation](https://ollycope.com/software/yoyo/latest/) for SQL
dependencies, authoring commands, and operator recovery commands. Use the
initializer above to keep schema creation and startup locking consistent.
