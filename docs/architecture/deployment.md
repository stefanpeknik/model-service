# How the chart deploys the catalog

The Helm chart in `deploy/helm/` reads the repo's `catalog/`
through its `catalog` symlink and renders one RayService, the control-plane Deployment
and the control plane's catalog ConfigMap. Nothing in the chart lists models or
analyses: it globs the catalog folders and the values pick from them.

## Values

| Key | What it sets |
| --- | --- |
| `environment` | The cluster this release deploys: the folder `deploy/helm/environments/<environment>/` whose `served-models/` patches are deep-merged over the base files and whose `workers/` files `workers` names. Required. |
| `servedModels` | `["*"]` deploys every file in `catalog/served-models/`, a list of IDs deploys those. Each one becomes a Serve application of the generic host, `model_service.served_model.serve:app`, at route `/<id>`. |
| `servedModelSettings` | What every served model gets, so catalog files don't repeat it: `cacheRoot` for TensorRT engines (ONNX Runtime keeps them in `<cacheRoot>/<id>/<digest>`), `env` for the replicas' environment variables (`MLFLOW_TRACKING_URI`, `HF_HOME`), and `setupTimeoutSeconds` for each application's `runtime_env`. |
| `analyses` | The same choice for `catalog/analyses/`. An analysis whose served model the release doesn't deploy fails the render. |
| `analysisRunner` | The one `analysis-runner` application (`model_service.analysis_runner.serve:app`, route `/analyses`) that runs every deployed analysis: its Serve deployment options, and as `userConfig` the threads reading tiles and the tiles in flight per run. It isn't rendered when no analysis is deployed. |
| `cpuWorkerResource` | A Ray custom resource only CPU worker groups advertise. The analysis runner requests it, so Ray never places it on a GPU worker's free CPUs. |
| `sharedStorageRoots` | The roots callers may name in shared-storage requests. |
| `workers`, `rayImage`, `rayGpuImage` | The worker groups, from `deploy/helm/environments/<environment>/workers/<name>.yaml`, and their images. |
| `controlPlane`, `empaia` | The control-plane Deployment and the EMPAIA provider's settings. |

`values.yaml` holds cluster-neutral defaults. Each cluster has a folder in
`environments/` with its `values.yaml` (installed with
`-f environments/<env>/values.yaml`), its served-model patches, its worker
groups and, in `k8s/`, the objects applied by hand before the first install
(`.helmignore` keeps them out of the chart):

- `rationai-test`: the test namespace beside the EMPAIA test instance
  ([runbook](../operations/rationai-test-ns.md));
- `sensitive-cluster`: `rationai-jobs-ns`, where model-service ran before this
  platform. Its values leave the image tags and the database and encryption
  Secrets empty, so the render fails naming each until they are filled in.

## From a served-model file to a Serve application

Each served-model file's blocks map onto Ray Serve's options:

- `resources` becomes `ray_actor_options` (`num_gpus`, `num_cpus`, `memory`).
- `batching` sets how the host batches tiles: a batch leaves when it holds `max_size` tiles or `wait_s` after its first one.
- `serving` sets the deployment's scaling and queue limits. Left out, they
  follow from `batching.max_size`: `max_ongoing_requests` is two batches,
  `max_queued_requests` sixteen, `target_ongoing_requests` half a batch, with
  0 to 4 replicas and Ray Serve's own `downscale_delay_s`. Keep
  `max_ongoing_requests` above `batching.max_size`, or batches never fill
  ([queues and backpressure](queues-and-backpressure.md)).
- `serving.max_concurrent_runs` is not a Serve option. It caps the scheduled
  analysis runs using the model at once, and the control plane reads it.

The chart's `_helpers.tpl` renders the application; the contract is written
out in `src/model_service/served_model/__init__.py`. The test cluster's patch for Virchow2, for
example, gives it half a GPU, smaller batches and one replica:

```yaml
# deploy/helm/environments/rationai-test/served-models/virchow2.yaml
batching: {max_size: 64, wait_s: 0.05}
resources: {gpus: 0.5, cpus: 1}
serving: {max_replicas: 1, target_ongoing_requests: 128, max_ongoing_requests: 256, max_queued_requests: 2048, downscale_delay_s: 300}
```

## The control plane's catalog

The catalog ConfigMap carries the deployed served models and analyses after
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
4 CPUs fits one served model that asks for `cpus: 4` and nothing else.
