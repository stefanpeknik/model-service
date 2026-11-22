# The rationai-infra-test-ns deployment

model-service beside the EMPAIA test instance, reading slides from the
instance's slide volume. It is the test deployment; the earlier one in
`peknik-ns` (with EMPAIA stand-ins) was removed on 2026-10-30. The namespace
and the EMPAIA services in it are shared: change only the objects listed here.
Release `model-service`, values `deploy/helm/rayservice/values.rationai-test.yaml`.

## Layout

- API Deployment `model-service-api` with PostgreSQL
  `model-service-postgres`.
- RayService with one CPU worker group (the analysis runner) and
  one MIG GPU worker group (served models) that scales to zero. The runner
  requests the `cpu_worker` Ray resource, which only the CPU group
  advertises, so it never takes the GPU worker's 4 CPUs that
  `prostate-classifier-1` needs (`cpuWorkerResource` in the chart's
  `values.yaml`). GPUs are scarce on the cluster: after testing, check that no
  `gpu-workers` pod remains (it goes about six minutes after the last request).
- Slides: linked through the direct-access route. The API and both worker groups mount
  the instance's `test-data-storage-pvc` read-only at `/data/Projects`, where
  its CDS plugins mount it, because model-service links the absolute path CDS
  answers. The claim is read-only on the volume itself, not only the mount,
  so kubelet never re-owns the EMPAIA files to model-service's fsGroup (the
  instance uses 1002, model-service 1000).
- Outputs volume `model-service-full-wsi-outputs` at `/mnt/shared` (the allowed
  shared-storage root) and scratch `model-service-full-wsi-scratch` at
  `/mnt/scratch`, which is also the EMPAIA workspace.
- App API: only `http://test-jobs-app-api:10009`.
- No marketplace registration: the instance's marketplace needs credentials
  model-service doesn't have yet.

Objects outside the chart, in `deploy/k8s/rationai-test-ns/`: PostgreSQL
and its claim, and the TensorRT and Hugging Face cache claims. Secrets: `model-service-db` (`postgres-password`, `database-url`,
`provider-context-key`, generated at install), `huggingface-secret` (the
namespace's own), `mlflow-public-credentials` (GPU workers, for MLflow
weights).

Namespace quota is 64 GiB of memory requests; the deployment uses about 31 GiB
idle and 49 GiB with the GPU worker. A RayService image or worker change starts
a second Ray cluster (about 4.5 GiB more) before removing the old one.

## Install

```sh
oc apply -f deploy/k8s/rationai-test-ns/storage.yaml -f deploy/k8s/rationai-test-ns/postgres.yaml
helm install model-service deploy/helm/rayservice -n rationai-infra-test-ns \
  -f deploy/helm/rayservice/values.rationai-test.yaml
```

Upgrades use `helm upgrade` with the same arguments.

Served models and analyses come from `catalog/`: the values deploy the IDs
named in `servedModels` (`prostate-classifier-1`, `virchow2` and
`prov-gigapath`; EPISeg's weights sit on an MLflow this cluster can't reach)
and in `analyses` (`prostate-heatmap`, `virchow2-embeddings` and
`prov-gigapath-embeddings`), with the `catalog/environments/rationai-test/`
patches. A catalog-only change needs no image rebuild and doesn't restart the API: kubelet
updates the mounted catalog ConfigMap (within about a minute) and the API
reloads it. If the new catalog doesn't load, the API keeps the old one and
`/healthz/catalog` answers 503 with the reason; the API log says the same.

## Ship code

Dependencies rarely change, so images are a thin code layer over the October
GPU and API images (the API base is still named `model-service-scheduler`). Pushing into the same Harbor repository reuses their
large base layers. Until a release of the EMPAIA SDK carries model-service's
additions, its wheel goes in `deploy/images/wheels/`.

```sh
REV=$(git rev-parse --short HEAD)
podman build -f deploy/images/Dockerfile.code-refresh \
  --build-arg BASE=cerit.io/stefanpeknik/model-service-workflow-gpu:checkpoint-20261008-bc6a23b \
  --build-arg PYTHON=/home/ray/anaconda3/bin/python \
  --build-arg 'EXTRA_PACKAGES=h5py>=3.11,<4 timm>=1.0,<2 pydantic-settings>=2.10.0,<3' \
  --build-arg SOURCE_REVISION=$REV \
  -t cerit.io/stefanpeknik/model-service-ray:core-$REV .
podman build -f deploy/images/Dockerfile.code-refresh \
  --build-arg BASE=cerit.io/stefanpeknik/model-service-scheduler:checkpoint-20261008-a91b519 \
  --build-arg RUN_USER=10001 --build-arg 'EXTRA_PACKAGES=lz4>=4.4.5 jsonschema>=4.23.0,<5 numpy imageio h5py pyyaml>=6.0.2' \
  --build-arg SOURCE_REVISION=$REV \
  -t cerit.io/stefanpeknik/model-service-api:core-$REV .
podman push cerit.io/stefanpeknik/model-service-ray:core-$REV
podman push cerit.io/stefanpeknik/model-service-api:core-$REV
```

Set both tags in `values.rationai-test.yaml`, then `helm upgrade`.

## Routing jobs here

The instance's JES reaches model-service through its `JES_HTTP_TARGETS`
entry `model-service`, set by hand on the `test-jobs-jes` Deployment to
`http://model-service-api.rationai-infra-test-ns.svc.cluster.local:8080/v1/empaia`.
A `helm upgrade` of the `test-jobs` release reverts it. Changing it touches a
shared service and needs authorization for the task. Releases before the
rename served it as `model-service-execution-scheduler`; the first upgrade
with the renamed chart needs this entry changed too.

## Try it

```sh
oc -n rationai-infra-test-ns port-forward svc/model-service-api 18080:8080
curl localhost:18080/v1/models
curl localhost:18080/healthz/deployments
curl localhost:18080/healthz/catalog
```

`prostate-classifier-1`, `virchow2` and `prov-gigapath` start the GPU
worker. To warm a served model with one blank tile, or start a job of the
prostate app from EMPAIA, run these scripts in an API pod:

```sh
oc exec -i deploy/model-service-api -- python - virchow2 224 < scripts/warm_model.py
oc exec -i deploy/model-service-api -- python - SLIDE_ID 120 < scripts/request_preprocessing.py
```

## Logs

- API: refused submissions and ignored EMPAIA configuration keys, with
  reasons.
- Analysis progress and errors: the analysis runner's JSON lines on the CPU
  worker pod's stdout (`oc logs`), each naming its execution, and the Ray
  dashboard's Serve page; model calls in the served model's replica log;
  Python tracebacks also in `/tmp/ray/session_latest/logs/worker-*.err`.
- EMPAIA side: `test-jobs-jes` (job start and end) and `test-jobs-app-api`
  (every App API call of a job).
