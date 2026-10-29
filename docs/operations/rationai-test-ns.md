# The rationai-infra-test-ns deployment

model-service beside the EMPAIA test instance, so it can read slides from the
instance's slide volume instead of downloading them. The namespace and the
EMPAIA services in it are shared: change only the objects listed here.
Release `model-service`, values `deploy/helm/rayservice/values.rationai-test.yaml`.

## Layout

Same images, models, builders and worker sizes as [peknik-ns](peknik-ns.md),
with these differences:

- Slides: `EMPAIA_SLIDE_ACCESS=direct`. The facade and both worker groups mount
  the instance's `test-data-storage-pvc` read-only at `/data/Projects`, where
  its CDS plugins mount it, because model-service links the absolute path CDS
  answers. The claim is read-only on the volume itself, not only the mount,
  so kubelet never re-owns the EMPAIA files to model-service's fsGroup (the
  instance uses 1002, model-service 1000).
- App API: only `http://test-jobs-app-api:10009`, no mock, no DNS search.
- No marketplace registration: the instance's marketplace needs credentials
  model-service doesn't have.
- Its own TensorRT and Hugging Face cache claims, which start empty.

Objects outside the chart, in `deploy/k8s/rationai-test-ns/`: PostgreSQL
`model-service-postgres` and its claim, the raw-inference claim, and the two
cache claims. Secrets: `model-service-db` (`postgres-password`, `database-url`,
`provider-context-key`, generated at install), `huggingface-secret` (the
namespace's own), `mlflow-public-credentials` (GPU workers, for MLflow
weights).

## Install

```sh
oc apply -f deploy/k8s/rationai-test-ns/storage.yaml -f deploy/k8s/rationai-test-ns/postgres.yaml
helm install model-service deploy/helm/rayservice -n rationai-infra-test-ns \
  -f deploy/helm/rayservice/values.rationai-test.yaml
```

Upgrades use `helm upgrade` with the same arguments. Images are built as for
peknik-ns.

## Routing jobs here

The instance's JES reaches model-service through its `JES_HTTP_TARGETS`
entry `model-service`. Pointing it at
`http://model-service-execution-scheduler.rationai-infra-test-ns.svc:8080`
changes a shared service and needs authorization for the task.
