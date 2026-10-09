# Simplified standalone workflow requests

Status: implemented and reviewed; final post-review local validation in progress.
No deployment.

## Agreed behavior

Public HTTP and SDK callers provide named inputs, input access, and an optional
`output_path`. Immediate requests include their deadline; scheduling derives it
from its timeout option. Processing configuration is no longer caller-controlled.
The old `resolved_configuration` field and other processing overrides are rejected.

The user explicitly chose execution-time settings, not admission-time snapshots.
The queue stores inputs, access, destination, and deadline. Each new standalone
invocation has `resolved_configuration: null`. The workflow application selects
defaults from its active catalog at the start of each attempt. Those settings
stay local to that attempt. A later retry may use newer settings.

The public request schema and internal invocation are separate. Input validation,
replay identity, deadlines, attempt-isolated outputs, polling, and cancellation
remain in their existing modules. Model-specific sampling and TIFF storage
defaults remain in the workflow registrations. Workflow-family consolidation
inspired by the legacy builder is a separate next step.

EMPAIA preparation is unchanged. Its existing provider-prepared settings and
previously stored resolved invocations still execute as before. No database schema
migration or queue deletion is needed. Old JSONB rows without `output_path` load
with the field's default. Existing admitted snapshots are not rewritten.

## Caller migration

Replace `resolved_configuration={...}` with an optional `output_path="..."`.
Omit stride and TIFF storage dimensions. Omitting the destination uses the
workflow's default filename; immediate calls with the same destination can replace
an earlier output. Read returned artifact paths for scheduled executions.

Both model-service and `../rationai-sdk-python` need updating together. Rebuild
public and workflow worker images before deployment. No images or cluster
resources have changed during this task. Existing uncommitted notebook/HTML
artifacts still contain the old call shape and need regeneration by their owner.
The Python demo request was migrated without staging its unrelated edits.

## Validation

Tests use the agreed public scheduler HTTP and SDK workflow methods, with fake
external Ray/model behavior and real generated TIFF/HDF5 files. Coverage includes
omitted settings, optional destinations, rejection of processing overrides,
admission/replay, and a settings change between queueing and processing.
The initial full working-tree run exposed one old-format resource fixture, now
migrated, and one unrelated failure: the uncommitted full-WSI worker preset sets
`minReplicas: 0`, while its deployment test still requires one. Those deployment
edits were preserved. A clean Git archive of implementation commit `5f0eed1`,
under `.cache/`, passed all 237 root tests with native image processing and isolated
localhost PostgreSQL 16. Tests imported source from the archive, not this dirty
checkout. No deployed test profile ran.

The SDK suite ran 14 tests: 13 passed and its unconfigured live integration test
was skipped. Ruff passed; focused strict mypy passed on ten model-service source
files and two SDK source files. UV used `.cache/uv` and the existing virtualenv.
The previously documented EMPAIA SDK 0.1.2 wheel remains installed; no `uv sync`
was run against the older sibling EMPAIA checkout.

## Standards

One low-severity documentation finding covered stale immediate-service and runtime
contracts. Updated descriptions distinguish public requests, deferred settings,
worker faults, and remote dispatch errors. No other documented-standard violations
were reported. The request conversion module and optional worker catalog were
considered appropriate; no design heuristic requires another change.

## Spec

One P2 finding concerned classifying missing worker settings as permanent caller
errors. It was reproduced through public HTTP for both a missing catalog and
registered required settings without defaults. Both now return retryable 503
`workflow_settings_unavailable`; invalid caller destinations remain permanent
validation errors. Resolution is a private runtime helper, separate from access
acquisition and execution. All other agreed behavior was reported as aligned.
