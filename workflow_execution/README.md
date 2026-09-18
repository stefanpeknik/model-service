# Workflow execution package

This package runs a declared workflow invocation and returns a normalized
workflow result. It contains no durable execution state and no provider finish
logic.

## Start here

- `contracts.py` contains the typed invocation, input, output, result, and
  failure values used by every workflow.
- `serialization.py` is the JSON transport codec for those values.
- `catalog.py` loads workflow definitions and validates their configuration.
- `application.py` validates an invocation, creates input access, enforces its
  deadline, runs the workflow, and validates the result.
- `shared_storage.py` implements standalone shared-storage input and output
  access.
- `../integrations/empaia/input_access.py` materializes EMPAIA inputs without
  coupling the core workflow runtime to the provider.
- `http.py` contains the workflow HTTP error envelope shared by ingress
  adapters.

`prostate_heatmap.py` is the workflow implementation.
`prostate_heatmap_serve.py` is its Ray Serve and FastAPI adapter. Deployment
configuration imports `workflow_execution.prostate_heatmap_serve:app`, so that
module remains a stable entry point.

Workflow code receives `WorkflowContext`. It does not inspect provider
credentials, schedule records, leases, or publication state. New input systems
implement the input-access interface; new scheduling providers implement the
separate interface in `execution_scheduler/providers.py`.
