# Validation SDK demo

Historical dummy-profile demo. Do not run `validation_sdk.py` unchanged against
the native full-WSI preset: its fake slide/mask inputs and inference routing
assumptions no longer match that deployment. Start at
[the documentation map](../docs/README.md) and
[the full-WSI runbook](../docs/operations/full-wsi-deployment.md) for current work.
The original instructions below are retained for a compatible dummy profile only.

This demo calls the deployed `model-service-validation` release in `peknik-ns`
through one public scheduler URL. It shows a durable scheduled workflow, an
immediate workflow, and raw classifier inference using the RationAI Python SDK.
It does not call a real EMPAIA installation.

## Before presenting

From the `model-service` repository, check that the scheduler is available and
the validation RayService is running:

```sh
oc -n peknik-ns get deployment model-service-validation-execution-scheduler
oc -n peknik-ns get rayservice model-service-validation
```

The deployment should report `1/1` ready, and the RayService should report
`Running`. The scheduler also forwards classifier calls to the existing
MLflow-backed RayService in `peknik-ns`. If that service or its GPU worker is
unavailable, the workflow portion can still run with `--part workflow`.

The script needs this repository's dependencies and the sibling SDK. If the
local `.venv` is missing or stale, prepare it once. Keep the uv cache inside
this repository:

```sh
UV_CACHE_DIR=.cache/uv uv sync --locked
UV_CACHE_DIR=.cache/uv uv pip install --python .venv/bin/python -e ../rationai-sdk-python
```

Keep a port-forward open in its own terminal:

```sh
oc -n peknik-ns port-forward svc/model-service-validation-execution-scheduler 18080:8080
```

## Run the demo

In another terminal, still from the `model-service` repository:

```sh
UV_CACHE_DIR=.cache/uv uv run --no-sync python demos/validation_sdk.py --origin http://127.0.0.1:18080
```

Use `--part workflow` or `--part inference` to show one part at a time. The
script requires `--origin` so the SDK never silently uses its default service
in another namespace.

The workflow input names two files already available on the validation shared
volume: `input/slide.dat` and `input/roi.dat`. Each invocation gets a unique
`output/demo-*.tiff` path. The script prints the scheduled execution ID and
status transitions, its result, the immediate result, and the classifier score.
If polling times out, keep the printed execution ID so you can inspect or
cancel that execution before submitting another.

## What this proves

The workflow application in this validation release uses a small processor
that writes an 8 by 8 grayscale TIFF. Its result demonstrates request
validation, queueing, execution, status polling, and output delivery. It is
not a clinical heatmap or a measure of model quality.

The inference example sends one all-zero 512 by 512 RGB tile to the
MLflow-backed `prostate-classifier-1` route. Its score proves that the SDK,
scheduler proxy, and model route work together. The number is not a meaningful
medical prediction. A cold GPU worker may take several minutes to start, so
allow up to 900 seconds for that request and rehearse before a live demo. Avoid
repeated inference requests just to keep the model warm.

The repeatable public-API tests behind this walkthrough are in
[`tests/integration/test_deployed_sdk.py`](../tests/integration/test_deployed_sdk.py).
The broader validation record and endpoint map are in
[`docs/operations/durable-workflow-execution.md`](../docs/operations/durable-workflow-execution.md).
