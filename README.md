# Model Service

Model deployment infrastructure for RationAI using Ray Serve on Kubernetes.

This repository contains:

- `builders/`: Legacy heatmap HTTP compatibility adapter, delegating to the workflow processor.
- `docker/`: Dockerfiles for building CPU and GPU environments.
- `docs/`: MkDocs documentation and architecture guides.
- `execution_scheduler/`: Durable admission, leases, retries, and provider finish.
  Start with
  [`execution_scheduler/README.md`](execution_scheduler/README.md).
- `helm/rayservice/`: A Helm chart that renders and deploys a KubeRay `RayService`.
- `integrations/`: External-system contracts and adapters, grouped by provider.
- `model_service/`: Public HTTP application, raw-inference forwarding, immediate
  execution, and dependency wiring. Start with
  [`model_service/README.md`](model_service/README.md).
- `models/`: Python entrypoints for model implementations (e.g., `binary_classifier.py`, `virchow2.py`).
- `workflow-definitions/`: Repository-owned workflow input and output contracts.
- `workflow_execution/`: Provider-neutral workflow execution and dispatch. Start with
  [`workflow_execution/README.md`](workflow_execution/README.md).
- `workflows/`: Built-in workflow registration and concrete processing pipelines.
  Start with [`workflows/README.md`](workflows/README.md).

## Documentation

Start at [the documentation map](docs/README.md) for the primary specification,
current ticket progress, implementation navigation, and validation evidence.
It distinguishes the standalone full-WSI checkpoint from unfinished EMPAIA and
failure-validation work, and identifies superseded plans and dummy-profile tests.

Model-only deployment and tuning references remain under `docs/guides/`.

## Quick Start (Kubernetes)

This example concerns the default model-inference chart. For the native workflow
checkpoint in `peknik-ns`, use [the full-WSI runbook](docs/operations/full-wsi-deployment.md).

Full walkthrough: `docs/get-started/quick-start.md`.

### Prerequisites

- Kubernetes cluster with KubeRay operator installed
- `kubectl` configured for the cluster
- `helm` installed locally

### Deploy

```bash
helm upgrade --install <release-name> helm/rayservice -n <namespace>
kubectl get rayservice <release-name> -n <namespace>
```

Use a dedicated test release name (for example `rayservice-model-my-model`) so local testing actions do not affect a shared main deployment.

### Access locally

```bash
kubectl port-forward -n <namespace> svc/<release-name>-serve-svc 8000:8000
```

### Test the reference model (`BinaryClassifier`)

The reference deployment in `helm/rayservice/applications/prostate-classifier-1.yaml` exposes an app at the route prefix:

- `/prostate-classifier-1`

`models/binary_classifier.py` expects a **request body that is LZ4-compressed raw bytes** of a single RGB tile:

- dtype: `uint8`
- shape: `(tile_size, tile_size, 3)`
- byte order: row-major (NumPy default)

Example (Python):

```bash
UV_CACHE_DIR=.cache/uv uv sync --locked
```

```python
import lz4.frame
import numpy as np
import requests

tile_size = 512  # must match RayService user_config.tile_size
tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)

payload = lz4.frame.compress(tile.tobytes())

resp = requests.post(
  "http://localhost:8000/prostate-classifier-1/",
  data=payload,
  headers={"Content-Type": "application/octet-stream"},
  timeout=60,
)
resp.raise_for_status()
print(resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text)
```

### Clean up

```bash
helm uninstall <release-name> -n <namespace>
```

## Support

- **Issues:** Report bugs or request features via [GitHub Issues](https://github.com/RationAI/model-service/issues)
- **Contact:** RationAI team at Masaryk University

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Authors

Developed and maintained by the RationAI team at Masaryk University, Faculty of Informatics.
