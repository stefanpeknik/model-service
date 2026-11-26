# How the chart deploys the catalog

The Helm chart in `deploy/helm/` reads the repo's `catalog/`
through its `catalog` symlink and renders one RayService, the control-plane Deployment
and the control plane's catalog ConfigMap. Nothing in the chart lists models or
pipelines: it globs the catalog folders and the values pick from them.

## Values

| Key | What it sets |
| --- | --- |
| `environment` | The cluster this release deploys: the folder `deploy/helm/environments/<environment>/` whose `models/` patches are deep-merged over the base files and whose `workers/` files `workers` names. Required. |
| `models` | `["*"]` deploys every file in `catalog/models/`, a list of IDs deploys those. Each one becomes a Serve application of the generic host, `model_service.model_host.serve:app`, at route `/<id>`. |
| `modelSettings` | What every model gets, so catalog files don't repeat it: `cacheRoot` for TensorRT engines (ONNX Runtime keeps them in `<cacheRoot>/<id>/<digest>`), `env` for the replicas' environment variables (`MLFLOW_TRACKING_URI`, `HF_HOME`), and `setupTimeoutSeconds` for each application's `runtime_env`. |
| `pipelines` | The same choice for `catalog/pipelines/`. A pipeline whose model the release doesn't deploy fails the render. |
| `pipelineRunner` | The one `pipeline-runner` application (`model_service.pipeline_runner.serve:app`, route `/pipelines`) that runs every deployed pipeline: its Serve deployment options, and as `userConfig` the threads reading tiles and the tiles in flight per run. It isn't rendered when no pipeline is deployed. |
| `cpuWorkerResource` | A Ray custom resource only CPU worker groups advertise. The pipeline runner requests it, so Ray never places it on a GPU worker's free CPUs. |
| `sharedStorageRoots` | The roots callers may name in shared-storage requests. |
| `workers`, `rayImage`, `rayGpuImage` | The worker groups, from `deploy/helm/environments/<environment>/workers/<name>.yaml`, and their images. |
| `storage` | The storage class of the claims the chart creates, the cache sizes, and with `outputSize` the shared outputs and scratch claims. |
| `postgres` | The scheduler's database, one per release: image, size, resources. |
| `secrets` | Secret values: the Hugging Face token, MLflow credentials, optionally the database password and provider-context key (generated otherwise). Never in a committed file. |
| `controlPlane`, `empaia` | The control-plane Deployment and the EMPAIA provider's settings. |

`values.yaml` holds cluster-neutral defaults. Each cluster has a folder in
`environments/` with its `values.yaml`, its model patches, its worker groups
and its `secrets.yaml`, which git ignores (the keys are in
`environments/secrets.example.yaml`). A release installs with both:
`helm upgrade --install model-service deploy/helm -f environments/<env>/values.yaml
-f environments/<env>/secrets.yaml`. The chart creates everything else the
release needs, PostgreSQL, the cache claims and the Secrets; the claims and
the generated database Secret survive `helm uninstall`
(`helm.sh/resource-policy: keep`). The database password and provider-context key
are generated on the first install and read back from the cluster on every
upgrade, so preview with `--dry-run=server`: `helm template` can't read them
and shows new ones.

- `rationai-test`: the test namespace beside the EMPAIA test instance
  ([runbook](../operations/rationai-test-ns.md));
- `sensitive-cluster`: `rationai-jobs-ns`, where model-service ran before this
  platform. Its values leave the image tags empty and its
  `secrets.yaml` needs the Hugging Face token, so the render fails naming each
  until they are filled in.

## From a model file to a Serve application

Each model file's blocks map onto Ray Serve's options:

- `resources` becomes `ray_actor_options` (`num_gpus`, `num_cpus`, `memory`).
- `batching` sets how the host batches tiles: a batch leaves when it holds `max_size` tiles or `wait_s` after its first one.
- `serving` sets the deployment's scaling and queue limits. Left out, they
  follow from `batching.max_size`: `max_ongoing_requests` is two batches,
  `max_queued_requests` sixteen, `target_ongoing_requests` half a batch, with
  0 to 4 replicas and Ray Serve's own `downscale_delay_s`. Keep
  `max_ongoing_requests` above `batching.max_size`, or batches never fill
  ([queues and backpressure](queues-and-backpressure.md)).
- `serving.max_concurrent_runs` is not a Serve option. It caps the scheduled
  pipeline runs using the model at once, and the control plane reads it.

The chart's `_helpers.tpl` renders the application; the contract is written
out in `src/model_service/model_host/__init__.py`. The test cluster's patch for Virchow2, for
example, gives it half a GPU, smaller batches and one replica:

```yaml
# deploy/helm/environments/rationai-test/models/virchow2.yaml
batching: {max_size: 64, wait_s: 0.05}
resources: {gpus: 0.5, cpus: 1}
serving: {max_replicas: 1, target_ongoing_requests: 128, max_ongoing_requests: 256, max_queued_requests: 2048, downscale_delay_s: 300}
```

## The caches only grow

The two cache claims (`storage.tensorrtCacheSize`, `storage.huggingfaceCacheSize`)
are written by model replicas and never pruned. Nothing in the chart or the
host deletes an old entry.

- `<cacheRoot>/<id>/<first 12 hex digits of the digest>` holds what a model's
  replicas build or fetch: ONNX Runtime's TensorRT engines and MLflow
  downloads. The folder is named after the model's digest, so any change the
  digest covers (weights, runtime, `pre`, `post`, `input`, `output`,
  `batching.max_size`, or how the digest itself is computed) starts a new
  folder and recompiles. The old folder stays behind.
- `HF_HOME` is the Hugging Face Hub cache, which keeps every revision it
  downloaded. Pinning a model to a new commit adds the new weights beside the
  old ones.

A full claim shows up when a replica starts: engines or weights can't be
written and the replica fails its setup. To free space, delete the folders
whose digest no running model has. `GET /<id>/describe` on a model's route
gives its full digest; the folder name is its first 12 characters. Wait
until a rollout has finished, because replicas of the previous version still
read their folders until they stop.

## The control plane's catalog

The catalog ConfigMap carries the deployed models and pipelines after
the environment's patches, as JSON. The control plane mounts it and resolves it through
the same `load_catalog` the catalog test uses. When kubelet updates the
mounted files (within about a minute of `helm upgrade`), the control plane reloads
them. A catalog that doesn't load leaves the previous one serving, and
`/healthz/catalog` answers 503 with the errors. A catalog change therefore
needs neither an image rebuild nor a control-plane restart; only components (code)
need a new image.

## Worker groups

Worker groups are defined in `deploy/helm/environments/<environment>/workers/` and selected
by `workers`. Kubernetes pod limits must exceed the Ray resources the
replicas on that pod reserve, plus Ray's own overhead: a GPU worker with
4 CPUs fits one model that asks for `cpus: 4` and nothing else.
