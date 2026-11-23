# Project context

Model-service runs **served models** and whole-slide **analyses**, both YAML
files in `catalog/` built from **components** in
`src/model_service/components/`. Start at
[docs/architecture/core.md](docs/architecture/core.md); terms are in
[docs/glossary.md](docs/glossary.md).

- Adding or changing a served-model file: follow
  [docs/guides/add-a-model.md](docs/guides/add-a-model.md).
- Adding or changing an analysis file, or its `expose.empaia`: follow
  [docs/guides/add-an-analysis.md](docs/guides/add-an-analysis.md).
- Writing or changing a component or a data format: follow
  [docs/guides/write-a-component.md](docs/guides/write-a-component.md).
- Looking up a component's options, a served model, an analysis or a format:
  read `docs/reference/`. It is generated; after changing components,
  `src/model_service/formats/` or `catalog/`, run
  `PYTHONPATH=src uv run scripts/generate.py` and commit what it writes
  (`tests/test_generated.py` fails until you do).
- Checking catalog files: `uv run pytest tests/catalog`.

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
