# Project context

For workflow, scheduling, EMPAIA, SDK, or deployment work, start at
[docs/README.md](docs/README.md). It routes to the primary grilling specification,
current progress, package maps, and dated evidence. Read only the references
needed for the task. September proposals and older sibling notes are history,
not a replacement specification.

Keep documentation aligned with changes. Update scratchpad progress when a ticket
or validation boundary changes; update its specification only for an agreed
decision. Record live validation separately from local test results.

Keep Git changes local; never push. Runtime operations need authorization for
the active task, not merely a command copied from an old runbook. Preserve
unrelated work and leave `models/` unchanged without explicit approval.

Use `uv` for Python and keep its cache in `.cache/uv`, never `/tmp`. Prepare
`.venv` with `uv sync` when needed. Keep credential files and their contents out
of documentation and tool output.
