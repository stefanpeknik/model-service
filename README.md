# Model Service

Model deployment infrastructure for RationAI using Ray Serve on Kubernetes.

This repository contains:

- `builders/`: WSI output aggregation services (e.g., `heatmap_builder.py`).
- `docker/`: Dockerfiles for building CPU and GPU environments.
- `docs/`: MkDocs documentation and architecture guides.
- `execution_scheduler/`: Durable admission, leases, retries, provider finish,
  and the execution HTTP interface. Start with
  [`execution_scheduler/README.md`](execution_scheduler/README.md).
- `helm/rayservice/`: A Helm chart that renders and deploys a KubeRay `RayService`.
- `integrations/`: External-system contracts and adapters, grouped by provider.
- `models/`: Python entrypoints for model implementations (e.g., `binary_classifier.py`, `virchow2.py`).
- `workflow-definitions/`: Repository-owned workflow input and output contracts.
- `workflow_execution/`: Provider-neutral workflow execution and Ray Serve
  applications. Start with
  [`workflow_execution/README.md`](workflow_execution/README.md).

## Documentation

- MkDocs content: `docs/`
- Key pages:
  - `docs/available-models.md`
  - `docs/get-started/quick-start.md`
  - `docs/guides/deployment-guide.md`
  - `docs/guides/adding-models.md`
  - `docs/guides/configuration-reference.md`
  - `docs/guides/troubleshooting.md`
  - `docs/architecture/overview.md`
  - `docs/architecture/request-lifecycle.md`
  - `docs/architecture/queues-and-backpressure.md`
  - `docs/architecture/batching.md`

## Quick Start (Kubernetes)

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
pip install numpy lz4 requests
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
