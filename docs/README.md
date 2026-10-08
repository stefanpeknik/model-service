# Start here

This is the documentation entry point for continuing the workflow-execution
work. It was consolidated on 2026-10-05. Check the branch and working tree before
assuming a dated document describes the source or deployment you are using.

## Requirements and progress

The primary source is the original grilling scratchpad:
[/home/stefanpeknik/.claude/.scratch/model-service/durable-workflow-execution](/home/stefanpeknik/.claude/.scratch/model-service/durable-workflow-execution/README.md).

- Read its [specification](/home/stefanpeknik/.claude/.scratch/model-service/durable-workflow-execution/spec.md)
  and the relevant numbered issue when implementing or changing behavior.
- Read its [progress](/home/stefanpeknik/.claude/.scratch/model-service/durable-workflow-execution/progress.md)
  for ticket status, unresolved findings, and validation still needed.
  The standalone full-WSI checkpoint is complete; the whole nine-ticket plan is not.
- Use source and configuration for actual behavior. If they disagree with the
  specification, report the discrepancy instead of silently changing requirements.

The scratchpad is outside this repository and is not included in a clone. If it
is unavailable, the package maps and dated evidence below remain useful, but
request the specification before making a scope or policy decision. Do not
substitute a September proposal or old EMPAIA note.

## Read by task

| Task | Read |
| --- | --- |
| Read function contracts and navigate through source | [Source docstrings guide](guides/docstrings.md), covering arguments, returns, exceptions, field meanings, and links to the package maps |
| Check the meaning of workflow, execution, provider, and artifact terminology | [Project glossary](../CONTEXT.md), which defines vocabulary rather than ticket status |
| Navigate public HTTP routes, raw inference, immediate execution, and application wiring | [Public application package map](../model_service/README.md) |
| Navigate scheduler ownership, transactions, retries, leases, and provider finish | [Scheduler package map](../execution_scheduler/README.md) |
| Change the execution database schema | [Yoyo migration guide](../execution_scheduler/persistence/migrations/README.md), including the WIP baseline and upgrade testing |
| Understand why durable scheduling runs outside Ray Serve | [Accepted scheduler placement decision](adr/0001-run-scheduler-outside-ray-serve.md), including the separate question of immediate HTTP routing |
| Navigate invocation types, input access, dispatch, and resource cleanup | [Workflow runtime package map](../workflow_execution/README.md) |
| Navigate heatmap processing and its Ray ingress | [Built-in workflows package map](../workflows/README.md) |
| Change named workflow inputs, outputs, or configuration | [Workflow definition guide](../workflow-definitions/README.md), then the definition and validator |
| Use public HTTP routes or understand the operating defaults | [Durable execution guide](operations/durable-workflow-execution.md), using its API guidance rather than its September deployment journal |
| Prepare or change the native GPU validation deployment | [Full-WSI runbook](operations/full-wsi-deployment.md), then `helm/rayservice/values.full-wsi.yaml` |
| Understand what the real WSI test proved | [2026-10-04 postmortem](operations/full-wsi-validation-postmortem-2026-10-04.md) |
| Validate the later local persistence and geometry fixes | [2026-10-05 regression report](reviews/checkpoint-fixes-2026-10-05.md) |
| Follow the final standalone rebuild and validation | [2026-10-06 validation record](operations/final-standalone-validation-2026-10-06.md), including the validated workflow checkpoint and remaining legacy-model limits |
| Change public SDK methods or types | [SDK workflows](../../rationai-sdk-python/docs/reference/resources/workflows.md), [executions](../../rationai-sdk-python/docs/reference/resources/executions.md), and [types](../../rationai-sdk-python/docs/reference/types/workflows.md) |
| Understand an earlier architecture choice | The historical references below, only as needed |
| Add or tune a low-level model deployment | [Model deployment guide](guides/deployment-guide.md) and [inference architecture](architecture/overview.md), not the workflow checkpoint runbook |

The package READMEs describe current ownership. The operational guide mixes
maintained semantics with dated observations; its September dummy-profile commands
are not instructions for the native full-WSI deployment.

The October 7 source reorganization moves public application code to
`model_service/`, persistence implementations to `execution_scheduler/persistence/`,
and heatmap code to `workflows/heatmap/`. Imports and deployment entry points have
been updated locally; tests and image rebuilds are deferred. Dated validation
reports describe their original revisions, not a validation of these moves.

The subsequent [October 7 ORM refactor](reviews/postgres-orm-2026-10-07.md)
replaces handwritten persistence SQL with typed SQLAlchemy ORM models. Local
PostgreSQL and application regression tests now cover the reorganized source
and ORM adapter. Neither change has been rebuilt into images or deployed.

The [October 8 library adoption review](reviews/library-adoption-2026-10-08.md)
records local Pydantic Settings, shared error-schema, PyJWT parsing, and offline
EAD schema-validation changes, the replacement of Alembic operations with Yoyo
migrations, their regression checks, and unimplemented EMPAIA client options.
Existing pre-Yoyo databases need a deliberate reset or verified adoption before
the new scheduler can start. It does not change ticket acceptance or
validate a new deployment.

## Validation boundaries

The October 4 live postmortem concerns the original warm GPU images. The October 5
regression report concerns later local code, not a new deployment. Its geometry
correction changes pixels on unaligned slides, so the old TIFF hash is not a
golden result for the corrected implementation.

Use returned artifact references. Scheduled output paths are isolated by attempt
and need not equal synchronous paths. Compare output contracts and artifact
content, not literal result dictionaries.

`demos/validation_sdk.py`, `tests/integration/test_deployed_sdk.py`, and the
sibling SDK's old deployed workflow smoke fixture use dummy-profile assumptions.
Do not run them against the native full-WSI preset unchanged. They are not a
replacement for the real-input validation in the postmortem.

EMPAIA is disabled in the full-WSI preset. Earlier mock tests and standalone
GPU success do not prove real EMPAIA interoperability. Consult scratchpad
progress before enabling it or planning failure injection. `/readyz` checks
durable API storage; `/healthz/workflows` reports Ray workflow availability
separately.

## Historical references

Keep these for rationale and evidence, not automatic session startup:

- [September proposal](architecture/workflow-execution-proposal.md) and
  [interface sketches](architecture/execution-interfaces.md). Their signatures
  and recovery details have been superseded by the specification and code.
- [Archived SQLite control plane](architecture/empaia-control-plane.md).
- `docs/reviews/` contains dated audits and resolutions. A finding is not still
  open merely because it appears in an old review; use scratchpad progress.
- [Sibling planning notes](../../notes/README.md) preserve earlier questions,
  rejected options, EMPAIA research, and [dated reports](../../notes/reports/README.md).
  Notes 14 through 17 are also intermediate history despite their current-sounding titles.
- [Sibling MkDocs experiment](../../documentation/README.md) is low-priority
  historical material. Its generated `site/` is not source. This is the directory
  formerly referred to as `../docs`; there is no sibling `docs/` directory now.

The old `/tmp/model-service-full-wsi-handoff-2026-10-02.md` is preparation history,
not a required starting point. Durable deployment and validation information is
in the runbook and postmortem linked above.

## Updating this map

Keep requirements in scratchpad `spec.md`, status and remaining work in its
`progress.md`, module ownership in the package READMEs, and measured outcomes in
dated reports. Add a link here when a new reference is needed; avoid another
competing overview or backlog. Label superseded instructions at their entry point
and retain useful original evidence.
