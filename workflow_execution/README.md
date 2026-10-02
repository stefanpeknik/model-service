# Workflow execution package

This package runs a declared workflow invocation and returns a normalized
workflow result. It contains no durable execution state and no provider finish
logic.

## Start here

- `contracts.py` contains the typed invocation, input, output, result, and
  failure values, with their structural validation rules.
- `serialization.py` converts those values for HTTP transport and JSONB
  persistence using Pydantic's contract schemas.
- `catalog.py` validates workflow declarations and requires configuration
  validation for every registered workflow.
- `registry.py` composes the built-in declarations, configuration rules, and
  workflow defaults. Adding YAML without registering its configuration fails
  at startup. Each workflow ID maps to one `WorkflowConfiguration` containing
  its validator and optional defaults. Configuration-free workflows register
  `validate_empty_configuration`.
- `application.py` validates an invocation, creates input access, enforces its
  deadline, runs the workflow, and validates the result.
- `shared_storage.py` implements standalone shared-storage input and output
  access.
- `../integrations/empaia/input_access.py` materializes EMPAIA inputs without
  coupling the core workflow runtime to the provider.
- `http.py` contains the workflow HTTP error envelope shared by ingress
  adapters.

`prostate_heatmap.py` is the workflow implementation.
It owns the prostate configuration schema and defaults. `heatmap.py` owns
slide tiling, prediction reconstruction, mask aggregation, and TIFF writing;
`ray_heatmap.py` supplies model predictions. The legacy
`builders/heatmap_builder.py` route delegates to that same slide processor.
`prostate_heatmap_serve.py` is its Ray Serve and FastAPI adapter. Deployment
configuration imports `workflow_execution.prostate_heatmap_serve:app`, so that
module remains a stable entry point.

`tests/test_workflow_deployment.py` checks declarations against the enabled
Helm applications. Deployment-file inspection is test-only; the runtime uses
the separate workflow health endpoint to inspect actual Ray routes.

Workflow code receives `WorkflowContext`. It does not inspect provider
credentials, schedule records, leases, or publication state. New input systems
implement the input-access interface; new scheduling providers implement the
separate interface in `execution_scheduler/providers.py`.

Input access implements `aclose`, including a no-op for shared storage. The
invocation deadline covers input acquisition and workflow computation. Cleanup
has a separate ten-second limit and runs under cancellation. A close coroutine
that exceeds that limit is cancelled and remains tracked until it stops. If
cleanup also fails after a workflow failure, the original failure determines
the result and retry policy. Failed or cancelled acquisition releases any
resources it created before returning an access object.

Scheduled shared-storage outputs live beneath
`.workflow-attempts/<attempt-hash>/` inside the supplied storage root. The
workflow result contains this actual artifact path. Each processing attempt
therefore writes its own files, so a late attempt cannot replace an accepted
artifact. Synchronous invocations without a processing attempt identifier keep
their configured relative output paths. These files remain available for
consumers after input access closes; this module does not delete completed
artifacts.
