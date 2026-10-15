# Project context

Start at [docs/architecture/core.md](docs/architecture/core.md) and
[docs/spec.md](docs/spec.md); [docs/README.md](docs/README.md) maps the rest.
`docs/history/` and older sibling notes describe the retired workflow design and
are not instructions.

Keep documentation aligned with changes. Record live validation separately from
local test results, in a dated file under `docs/operations/`. Tests need
`TEST_POSTGRES_DSN` for the PostgreSQL suite (see README).

Keep Git changes local; never push. Runtime operations need authorization for
the active task, not merely a command copied from an old runbook. Preserve
unrelated work and leave `src/model_service/models/` unchanged without explicit approval.

Use `uv` for Python and keep its cache in `.cache/uv`, never `/tmp`. Prepare
`.venv` with `uv sync` when needed. Keep credential files and their contents out
of documentation and tool output.
