# Project context

Model-service runs **models** and whole-slide **pipelines**, both YAML
files in `catalog/` built from **components** in `src/components/`.
`src/model_service/` is the plumbing that loads, checks and runs them. Start at
[docs/architecture/core.md](docs/architecture/core.md); terms are in
[docs/glossary.md](docs/glossary.md).

- Adding or changing a model file: follow
  [docs/guides/add-a-model.md](docs/guides/add-a-model.md).
- Adding or changing a pipeline file, or its `providers.empaia`: follow
  [docs/guides/add-a-pipeline.md](docs/guides/add-a-pipeline.md).
- Writing or changing a component or a data format: follow
  [docs/guides/write-a-component.md](docs/guides/write-a-component.md).
- Looking up a component's options, a model, a pipeline or a format:
  read `docs/reference/`. It is generated; after changing components,
  `src/model_service/formats/` or `catalog/`, run
  `PYTHONPATH=src uv run tools/generate.py` and commit what it writes
  (`tests/test_generated.py` fails until you do).
- Checking catalog files: `uv run pytest tests/catalog`. Before deploying a
  component or catalog change, run the whole suite: `uv run pytest`.

`docs/history/` and the dated `docs/operations/` records describe earlier
designs and the plans that replaced them. They are records, not instructions.

Keep documentation aligned with changes. Record live validation separately from
local test results, in a dated file under `docs/operations/`. Tests need
`TEST_POSTGRES_DSN` for the PostgreSQL suite (see README).

Keep Git changes local; never push. Runtime operations need authorization for
the active task, not merely a command copied from an old runbook. Preserve
unrelated work.

Use `uv` for Python and keep its cache in `.cache/uv`, never `/tmp`. Prepare
`.venv` with `uv sync` when needed. Keep credential files and their contents out
of documentation and tool output.
