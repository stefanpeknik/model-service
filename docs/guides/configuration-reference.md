# Configuration Reference

This guide documents the current Helm-based configuration model for Model Service.

## 1. Configuration Layout

Model Service uses Helm chart files in `deploy/helm/rayservice/` and the repo's `catalog/`, which the chart reads through its `catalog` symlink:

- `catalog/served-models/<id>.yaml`: one file per served model: its weights, runtime, input and output, batching, resources and serving limits. The file name is the model ID. Every file becomes one Ray Serve application of the generic host.
- `catalog/environments/<environment>/`: patches deep-merged over the base files for the environment the values file names in `environment`.
- `servedModels:` and `servedModelSettings:` in the values file: which served models deploy (`["*"]` or named IDs), and settings every served model gets (TensorRT cache root, replica environment variables, `runtime_env` setup timeout).
- `analyses:` and `analysisRunner:` in the values file: which `catalog/analyses/` files deploy (`["*"]` or named IDs), and the size of the one analysis-runner application that runs them; it is left out when no analysis deploys.
- `deploy/helm/rayservice/values.yaml`: Chart values and shared RayService settings.
- `deploy/helm/rayservice/workers/`: Worker group definitions (CPU/GPU pools).
- `deploy/helm/rayservice/templates/rayservice.yaml`: Template that renders RayService with combined `serveConfigV2`.

## 2. Served-Model Files

```yaml
# catalog/served-models/prostate-classifier-1.yaml
weights: {use: mlflow, uri: "mlflow-artifacts:/65/…/prostate_model_norm.onnx"}
runtime: {use: onnx, tensorrt: true, workspace_gib: 8, optimization_level: 3, threads: 4}
input: {tile_size: 512, mpp: 0.468}
output: {kind: scores, channels: 1, size: 1, range: [0, 1]}
batching: {max_size: 256, wait_s: 0.05}
resources: {gpus: 1, cpus: 4, memory_gib: 12}
```

Field summary:

- `weights`: where the weights come from (`mlflow`, `huggingface` with a pinned commit, `local`).
- `runtime`: what runs them (`onnx`, `timm`) and its options.
- `pre`, `post`: steps before and after the runtime, e.g. `[{use: cls_mean_pool, registers: 4}]`.
- `input`: tile size in pixels and its resolution (`mpp`).
- `output`: `{kind: scores, …}` or `{kind: embedding, summaries: […]}`.
- `batching`: the largest batch and how long to wait for one to fill.
- `resources`: GPUs, CPUs and memory per replica.
- `serving`: scaling and queue limits (below).

The chart renders each file as one application (`model_service.served_model.serve:app`, route `/<id>`, one `ServedModel` deployment); `src/model_service/served_model/__init__.py` lists exactly what it renders.

## 3. Serving Limits

`serving:` sets the Ray Serve deployment options. Left out, they follow from `batching.max_size`:

- `max_ongoing_requests`: requests one replica takes at once; two batches.
- `max_queued_requests`: requests queued before callers are refused; sixteen batches.
- `min_replicas`, `max_replicas`: 0 and 4.
- `target_ongoing_requests`: requests per replica the autoscaler aims at; half a batch.
- `downscale_delay_s`: Ray Serve's default (600) unless given.

`resources:` becomes `ray_actor_options` (`num_gpus`, `num_cpus`, `memory`).

The test cluster's patch for prostate, for example, runs one replica on half a GPU (its `weights:` line left out):

```yaml
# catalog/environments/rationai-test/served-models/prostate-classifier-1.yaml
batching: {max_size: 64}
resources: {gpus: 0.5, cpus: 1}
serving: {max_replicas: 1, target_ongoing_requests: 128, max_ongoing_requests: 256, max_queued_requests: 2048, downscale_delay_s: 300}
```

Practical guidance:

- Lower `target_ongoing_requests` for latency-sensitive or compute-heavy models.
- Keep `downscale_delay_s` higher to avoid frequent scale up/down oscillation.
- Keep `max_ongoing_requests` above `batching.max_size`, or batches never fill (see [Queues and backpressure](../architecture/queues-and-backpressure.md)).

## 4. Cluster and Worker Resources

Worker groups are defined under `deploy/helm/rayservice/workers/` and referenced by chart values.

### Default Worker Profiles (Recommended)

For most use cases, keep the chart defaults and only tune application/deployment-level settings.

- `cpu-workers`: default CPU worker profile.
- `mig20-workers`: default GPU worker profile (MIG 2g.20gb).

These defaults are selected in `deploy/helm/rayservice/values.yaml` and rendered from `deploy/helm/rayservice/workers/*.yaml`.

Customize worker templates only when you need specific resources or scheduling behavior (for example different GPU class, memory envelope, node selectors, tolerations, or custom network/proxy constraints).

Example worker group:

```yaml
- groupName: cpu-workers
  replicas: 1
  minReplicas: 0
  maxReplicas: 10
  template:
    spec:
      containers:
        - name: ray-worker
          resources:
            requests:
              cpu: "4"
              memory: "8Gi"
            limits:
              cpu: "8"
              memory: "16Gi"
```

Sizing rule of thumb:

- Physical pod limits in Kubernetes must exceed total logical actor reservations plus Ray overhead.

## 5. Working Directory Cache Note

Ray caches `runtime_env.working_dir` by URL string. If code was updated but URL is unchanged, an older cached snapshot may be reused.

Cache-busting example:

```yaml
runtime_env:
  working_dir: https://github.com/RationAI/model-service/archive/refs/heads/main.zip?v=2
```

For troubleshooting details, see [Troubleshooting](troubleshooting.md).

## Related Guides

- [Deployment Guide](deployment-guide.md)
- [Adding New Models](adding-models.md)
- [Optimization Guide](optimization-guide.md)
- [Troubleshooting](troubleshooting.md)
- [Architecture Overview](../architecture/overview.md)
