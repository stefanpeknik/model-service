# Simplified standalone workflow requests

Status: implemented; final local validation and review in progress. No deployment.

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
Final results and two-axis review will be recorded here before closure.
