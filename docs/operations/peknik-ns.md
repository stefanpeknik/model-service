# The peknik-ns test deployment

A playground: its data is disposable. Release `model-service-validation`,
values `helm/rayservice/values.peknik.yaml`. Use the namespace's `kubectl`
(`.cache/deploy-tools/kubectl` in the main checkout) with context `kuba-cluster`.

## Layout

- Facade Deployment `model-service-validation-execution-scheduler` with
  PostgreSQL `model-service-validation-postgres` (secret
  `model-service-validation-db`).
- RayService with one CPU worker group (builders, the constant model) and one
  MIG GPU worker group that scales to zero. GPUs are scarce on the cluster:
  after testing, check that no `gpu-workers` pod remains (it goes about six
  minutes after the last request).
- Shared volume `model-service-validation-full-wsi-outputs`, mounted at
  `/mnt/shared` on the facade and every worker. Test inputs are under
  `/mnt/shared/inputs/prostate-wsi/`; the allowed shared-storage root is
  `/mnt/shared`.
- Namespace quota is 40 GiB of memory requests. A RayService image or worker
  change starts a second Ray cluster before removing the old one; the new
  workers may wait a few minutes for quota while the old ones drain.

## Ship code

Dependencies rarely change, so images are a thin code layer over the October
GPU and facade images. Pushing into the same Harbor repository reuses their
large base layers.

```sh
REV=$(git rev-parse --short HEAD)
podman build -f docker/Dockerfile.code-refresh \
  --build-arg BASE=cerit.io/stefanpeknik/model-service-workflow-gpu:checkpoint-20261008-bc6a23b \
  --build-arg PYTHON=/home/ray/anaconda3/bin/python \
  --build-arg 'EXTRA_PACKAGES=h5py>=3.11,<4 timm>=1.0,<2' \
  --build-arg SOURCE_REVISION=$REV \
  -t cerit.io/stefanpeknik/model-service-workflow-gpu:core-$REV .
podman build -f docker/Dockerfile.code-refresh \
  --build-arg BASE=cerit.io/stefanpeknik/model-service-scheduler:checkpoint-20261008-a91b519 \
  --build-arg RUN_USER=10001 --build-arg SOURCE_REVISION=$REV \
  -t cerit.io/stefanpeknik/model-service-scheduler:core-$REV .
podman push cerit.io/stefanpeknik/model-service-workflow-gpu:core-$REV
podman push cerit.io/stefanpeknik/model-service-scheduler:core-$REV
```

Set both tags in `values.peknik.yaml`, then:

```sh
helm upgrade model-service-validation helm/rayservice -n peknik-ns \
  --reset-values --force-conflicts -f helm/rayservice/values.peknik.yaml
```

`--force-conflicts` is needed while fields set by earlier `kubectl patch`
commands are still owned by them. A field that another manager owns and Helm no
longer renders stays on the object; if that happens to the facade, delete its
Deployment and upgrade again.

## Try it

```sh
kubectl -n peknik-ns port-forward svc/model-service-validation-execution-scheduler 18080:8080
curl localhost:18080/v1/models
curl localhost:18080/healthz/deployments
```

`constant-classifier` exercises the heatmap builder and the scheduler on the CPU
worker only. `prostate-classifier-1` and `virchow2` start the GPU worker.
