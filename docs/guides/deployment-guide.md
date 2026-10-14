# Deployment Guide

This guide focuses on one goal: get a model running on Kubernetes through the Helm chart, verify it, and update it safely.

## What This Guide Covers

- Deploying a model after it has been exported to ONNX, uploaded to MLflow, and wired into a Python entrypoint.
- Deploying a model from `helm/rayservice/models/`.
- Monitoring rollout health.
- Running a quick endpoint smoke test.
- Updating code/config/model artifact without breaking traffic.
- Recovering from failed changes.

## What Is Covered Elsewhere

To avoid duplication, deeper topics are documented in dedicated guides:

- Exporting to ONNX, uploading to MLflow, and Python model class structure: [Adding New Models](adding-models.md)
- All YAML knobs and autoscaling fields: [Configuration Reference](configuration-reference.md)
- Incident diagnosis and runtime failure patterns: [Troubleshooting](troubleshooting.md)

## Prerequisites

Before deployment, confirm:

- KubeRay operator is installed and healthy.
- You can deploy into your target namespace (for example `rationai-jobs-ns`).
- Your model entrypoint is importable as `module.path:app`.
- Cluster can access remote dependencies (for example MLflow, object storage, or GitHub `working_dir`).

## Step 1: Prepare the Python Entry Point

If you still need to export the model or upload it to MLflow, do that first in [Adding New Models](adding-models.md). Once the artifact exists, create the model class in `models/` and wire `__init__`, `reconfigure`, `predict`, and `root` as described there.

## Step 2: Create or Update App Definition

Create a file in `helm/rayservice/models/` (for example `my-model.yaml`). The
`serve` block is a Ray Serve application and is passed to Ray unchanged; the
`inference` block feeds the facade's model catalog:

```yaml
inference:
  contract: spatial-scores      # or tile-embedding (then also mpp, summaries)
serve:
  name: my-model
  import_path: models.my_model:app
  route_prefix: /my-model       # must be /<name>
  deployments:
    - name: MyModel
      autoscaling_config:
        min_replicas: 0
        max_replicas: 4
      user_config:
        tile_size: 512
        output_tile_size: 1
        n_channels: 1
        mpp: 0.468
        model:
          _target_: providers.model_provider:mlflow
          artifact_uri: mlflow-artifacts:/...
```

- Add `my-model` to `helm/rayservice/values.yaml` under `models`.

Notes:

- Deployed model files run the code baked into the image. While developing a
  new implementation you may add `runtime_env.working_dir` with a branch ZIP to
  the `serve` block; remove it before merging so model and builder code cannot
  drift apart.
- If code changed but the ZIP URL did not, append a cache-busting suffix (for example `?v=2`).
- Keep advanced tuning in YAML, but use [Configuration Reference](configuration-reference.md) as the source of truth for field meanings.

## Step 3: Deploy With Helm

```bash
helm upgrade --install <release-name> helm/rayservice -n rationai-jobs-ns
```

Use a dedicated release name while testing (for example `rayservice-model-my-model`).

## Step 4: Watch Rollout

```bash
kubectl get rayservice <release-name> -n rationai-jobs-ns -w
kubectl get pods -n rationai-jobs-ns -l ray.io/cluster=<release-name>
```

If rollout stalls, inspect head logs:

```bash
kubectl logs -n rationai-jobs-ns -l ray.io/node-type=head --tail=200
```

## Step 5: Smoke Test Endpoint

Port-forward Serve service:

```bash
kubectl port-forward -n rationai-jobs-ns svc/<release-name>-serve-svc 8000:8000
```

Send one compressed request:

```python
import lz4.frame
import numpy as np
import requests

tile = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
payload = lz4.frame.compress(tile.tobytes())

resp = requests.post(
    "http://localhost:8000/my-model/",
    data=payload,
    headers={"Content-Type": "application/octet-stream"},
)
resp.raise_for_status()
print(resp.text)
```

Additionally, you can use the SDK client to send requests with the same payload format as your application expects.

## Step 6: Update Safely

### Update code

1. Push code changes to the branch used in `runtime_env.working_dir`.
2. Redeploy:

```bash
helm upgrade --install <release-name> helm/rayservice -n rationai-jobs-ns
```

3. If old code is still used, bump `working_dir` cache key (`?v=<n>`) and deploy again.

### Update config

1. Edit application YAML (for example autoscaling, queue limits, `user_config`).
2. Redeploy with the same Helm command.

### Update model weights (MLflow)

Point `artifact_uri` in `user_config` to the new artifact version and redeploy.

## Rollback and Recovery

RayService is a CRD, so typical Kubernetes rollout commands are not the main recovery path. Use state inspection and re-apply a known-good spec.

```bash
kubectl get rayservice <release-name> -n rationai-jobs-ns -o yaml
kubectl describe rayservice <release-name> -n rationai-jobs-ns
kubectl logs -n rationai-jobs-ns -l ray.io/node-type=head --tail=200
```

## Common Pitfalls

### Pod resources vs replica resources

A deployment requesting `num_cpus: 4` cannot schedule onto workers that expose fewer allocatable CPUs. If replicas remain pending, verify worker pod requests/limits and Ray actor reservations match.

### Runtime environment cache

Ray caches `runtime_env.working_dir` by URL string. If URL stays identical, code refresh may not happen. Use a version suffix in URL to force refresh.

## Multi-Model Deployment

You can deploy multiple models by adding multiple files in `helm/rayservice/models/` and listing them under `models`. Helm renders them, plus one application per builder, into one `serveConfigV2`.

Keep routes unique and validate each endpoint separately after deploy.

## Related Guides

- [Adding New Models](adding-models.md)
- [Configuration Reference](configuration-reference.md)
- [Optimization Guide](optimization-guide.md)
- [Troubleshooting](troubleshooting.md)
- [Architecture Overview](../architecture/overview.md)
