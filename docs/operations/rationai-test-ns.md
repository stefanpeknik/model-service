# The rationai-infra-test-ns deployment

model-service beside the EMPAIA test instance, reading slides from the
instance's slide volume. It is the test deployment; the earlier one in
`peknik-ns` (with EMPAIA stand-ins) was removed on 2026-10-30. The namespace
and the EMPAIA services in it are shared: change only the objects listed here.
Release `model-service`, values `deploy/helm/rayservice/values.rationai-test.yaml`.

## Layout

- Facade Deployment `model-service-execution-scheduler` with PostgreSQL
  `model-service-postgres`.
- RayService with one CPU worker group (builders, the constant model) and one
  MIG GPU worker group that scales to zero. GPUs are scarce on the cluster:
  after testing, check that no `gpu-workers` pod remains (it goes about six
  minutes after the last request).
- Slides: `EMPAIA_SLIDE_ACCESS=direct`. The facade and both worker groups mount
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
and its claim, the raw-inference claim, and the TensorRT and Hugging Face cache
claims. Secrets: `model-service-db` (`postgres-password`, `database-url`,
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

## Ship code

Dependencies rarely change, so images are a thin code layer over the October
GPU and facade images. Pushing into the same Harbor repository reuses their
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
  --build-arg RUN_USER=10001 --build-arg 'EXTRA_PACKAGES=lz4>=4.4.5 jsonschema>=4.23.0,<5 numpy imageio h5py' \
  --build-arg SOURCE_REVISION=$REV \
  -t cerit.io/stefanpeknik/model-service-scheduler:core-$REV .
podman push cerit.io/stefanpeknik/model-service-ray:core-$REV
podman push cerit.io/stefanpeknik/model-service-scheduler:core-$REV
```

Set both tags in `values.rationai-test.yaml`, then `helm upgrade`.

## Routing jobs here

The instance's JES reaches model-service through its `JES_HTTP_TARGETS`
entry `model-service`, set by hand on the `test-jobs-jes` Deployment to
`http://model-service-execution-scheduler.rationai-infra-test-ns.svc.cluster.local:8080/v1/empaia`.
A `helm upgrade` of the `test-jobs` release reverts it. Changing it touches a
shared service and needs authorization for the task.

## Try it

```sh
oc -n rationai-infra-test-ns port-forward svc/model-service-execution-scheduler 18080:8080
curl localhost:18080/v1/models
curl localhost:18080/healthz/deployments
```

`constant-classifier` exercises the heatmap builder and the scheduler on the CPU
worker only. `prostate-classifier-1` and `virchow2` start the GPU worker.

Scheduled tiles are cheap to test on `constant-classifier` (128x128 tiles):

```sh
uv run python -c "import lz4.frame, sys; sys.stdout.buffer.write(lz4.frame.compress(bytes(128*128*3)))" > tile.lz4
curl -s -XPOST localhost:18080/v1/models/constant-classifier/schedule \
  -H 'Idempotency-Key: try-1' --data-binary @tile.lz4
curl -s localhost:18080/v1/executions/<id>/result
```

## Logs

- Facade: refused submissions and ignored EMPAIA configuration keys, with
  reasons.
- Builder progress and errors: the builder replica's log
  (`/tmp/ray/session_latest/logs/serve/replica_builder-heatmap_*.log` on the CPU
  worker, or the Ray dashboard's Serve page); model calls in the model
  replica's log; Python tracebacks also in `/tmp/ray/session_latest/logs/worker-*.err`.
- EMPAIA side: `test-jobs-jes` (job start and end) and `test-jobs-app-api`
  (every App API call of a job).
