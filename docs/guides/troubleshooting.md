# Troubleshooting

This page lists the most common issues when deploying and running models in Model Service (Ray Serve on KubeRay).

## Quick Triage Checklist

Start here before digging deeper:

Use your RayService name from the Helm install (preferably a dedicated test release, e.g. `rayservice-model-<test>`) in the commands below.

```bash
kubectl get rayservice -n rationai-jobs-ns
kubectl describe rayservice <release-name> -n rationai-jobs-ns
kubectl get pods -n rationai-jobs-ns
```

Then inspect logs:

```bash
kubectl logs -n rationai-jobs-ns -l ray.io/node-type=head --tail=200
kubectl logs -n rationai-jobs-ns -l ray.io/node-type=worker --tail=200
```

## Check Rancher (cluster events)

If a server request is **timing out** or you see resources exhausted, the Rancher cluster UI often contains useful cluster-level events and node/pod status that explain the root cause. Visit the RayService explorer for the `rationai-jobs-ns` namespace:

[Rancher RayService Explorer — rationai-jobs-ns / rayservice-model](https://rancher.cloud.trusted.e-infra.cz/dashboard/c/c-m-lr2wzbk8/explorer/ray.io.rayservice/rationai-jobs-ns/rayservice-model)

Look for events, node capacity, and pod scheduling failures — these are frequently the reason Serve endpoints drop or time out.

Example:
```yaml
virchow2:
        serveDeploymentStatuses:
          Virchow2:
            message: >-
              Deployment 'Virchow2' in application 'virchow2' has 1 replicas
              that have taken more than 30s to be scheduled. This may be due to
              waiting for the cluster to auto-scale or for a runtime environment
              to be installed. Resources required for each replica: {"CPU": 4.0,
              "GPU": 1.0, "memory": 8589934592}, total resources available:
              {"memory": 4294967296.0}. Use `ray status` for more details.
            status: UPSCALING
        status: RUNNING
```
There are no resources available to create a worker (4 CPUs, 1 GPU and 8GB memory) so the deployment is stuck in `UPSCALING` until the resources are available.
## RayService Shows `DEPLOY_FAILED`

### What it usually means

Ray Serve could not start the application or deployment. The root cause is typically visible in the Ray Serve controller logs.

### What to do

1. Describe the RayService for events:

```bash
kubectl describe rayservice <release-name> -n rationai-jobs-ns
```

2. Open the Ray dashboard (helps with Serve deployment errors):

```bash
kubectl port-forward -n rationai-jobs-ns svc/<release-name>-head-svc 8265:8265
```

Visit `http://localhost:8265`.

3. Look for Python import errors / missing dependencies:

```bash
kubectl logs -n rationai-jobs-ns -l ray.io/node-type=worker --tail=500
```

## ImportError / ModuleNotFoundError

### Symptoms

- Serve deployment fails immediately.
- Logs show `ModuleNotFoundError: No module named ...`.

### Causes

- Dependency not installed in the runtime environment.
- Wrong `import_path`.
- `working_dir` does not contain the expected code.

### Fix

- Ensure `import_path` matches your file:
  - Example: `model_service.models.binary_classifier:app` means there is `src/model_service/models/binary_classifier.py` defining `app = ...`.
- Add missing dependencies to `runtime_env.pip`.

In this repository, dependencies are typically installed per deployment:

```yaml
deployments:
  - name: BinaryClassifier
    ray_actor_options:
      runtime_env:
        pip: ["onnxruntime>=1.23.2", "mlflow<3.0", "lz4>=4.4.5"]
```

## Worker Crashes (OOMKilled)

### Symptoms

- Pods in `kubectl get pods` show status `OOMKilled` or high restart counts.
- `kubectl describe pod ...` shows "Last State: Terminated (Reason: OOMKilled)".
- Ray Dashboard shows unexpected actor deaths.

### Causes

- The model loaded into memory + the input batch size exceeds the container's memory limit.
- **Physical vs Logical Mismatch**: Ray was told the actor needs 2GB, so it scheduled it on a node, but the actual Python process used 4GB, causing Kubernetes to kill it.

### Fix

You must increase **both** the Ray logical allocation and the Kubernetes physical limit.

1. Increase `ray_actor_options.memory` (Software limit):

   ```yaml
   ray_actor_options:
     memory: 4294967296 # 4 GiB
   ```

2. Increase Kubernetes container limits (Hardware limit):
   Ensure the worker configurations in your Helm values (`deploy/helm/rayservice/values.yaml` or relevant worker definitions) provide **more** memory than the sum of all actors on that node plus overhead (~30%).

   ```yaml
   resources:
     limits:
       memory: "6Gi" # Must cover the 4GB actor + Ray overhead
   ```

## Autoscaling Not Working (Replicas Don’t Change)

### Serve replicas not scaling

Check your deployment has autoscaling configured:

```yaml
autoscaling_config:
  min_replicas: 0
  max_replicas: 4
  target_ongoing_requests: 32
```

Also note:

- Scale up/down is not instantaneous (delays and smoothing apply).
- If traffic is low, you may stay at `min_replicas`.

### Worker pods not scaling

Worker pod scaling requires cluster autoscaling enabled:

```yaml
rayClusterConfig:
  enableInTreeAutoscaling: true
  autoscalerOptions:
    idleTimeoutSeconds: 60
```

Also ensure `workerGroupSpecs[*].minReplicas/maxReplicas` allow scaling.

## Not Enough CPU / Memory (Pods Pending)

### Symptoms

- Pods stay in `Pending`.
- Events mention `Insufficient cpu` or `Insufficient memory`.

### Fix

1.  **Check Physical vs Logical**:
    - _Physical_: Can K8s schedule the pod? `kubectl describe pod` will show if nodes are full.
    - _Logical_: Can Ray schedule the actor? Check `ray status` or the dashboard. Ray might say "0/X CPUs available" even if the pod exists, because other actors consumed the slots.

2.  **Adjust Resources**:
    - Reduce per-replica requirements (`ray_actor_options.num_cpus`, `memory`).
    - Increase cluster capacity (maxReplicas) or per-worker limits.

Inspect pod scheduling events:

```bash
kubectl describe pod <pod-name> -n rationai-jobs-ns
```

## MLflow / Artifact Download Problems

### Symptoms

- `mlflow.artifacts.download_artifacts` fails.
- Timeouts during replica initialization.

### Fix

- Ensure `MLFLOW_TRACKING_URI` is set and reachable from the cluster.
- Ensure the cluster has network access (proxy settings if needed).
- Verify the `artifact_uri` exists and permissions are correct.

In your model's Helm YAML definition this is typically configured via `env_vars`:

```yaml
ray_actor_options:
  runtime_env:
    env_vars:
      MLFLOW_TRACKING_URI: http://mlflow.rationai-mlflow:5000
```

## Code Updates Not Applying (Working Dir Cache)

### Symptoms

- You updated your model Python code, pushed it to GitHub, and ran `helm upgrade`, but Ray keeps deploying the old logic or throws errors that were already fixed.

### Cause

Ray downloads the source code defined in `working_dir`: `https://github.com/.../main.zip` and saves it strictly based on the URL string constraint. If the URL hasn't changed, Ray Serve will NOT re-download the archive, effectively clinging to an old snapshot.

### Fix

Append a cache buster query parameter directly to your `working_dir` setup:

```yaml
runtime_env:
  config:
    setup_timeout_seconds: 1800
  working_dir: https://github.com/RationAI/model-service/archive/refs/heads/main.zip?v=1
```

Whenever you push subsequent revisions, just manually bump the `v=1` to `v=2`. During the next Helm deployment process, Ray evaluates the URL as new, retrieves the fresh code zip, and deploys successfully.

## Large ONNX Model Export (>2GB)

### Problem

When exporting a large PyTorch model (>2GB) to ONNX, the resulting file is split into multiple parts: a `.onnx` graph file plus external weight files (`.bin` or `.pb`). Deploying such multi-part exports can be tricky.

### Two Approaches

**Option 1: Merge into a single ONNX file**

Load the model with external data and save it back as a single unified file:

```python
import onnx

model = onnx.load("model-with-external-data.onnx", load_external_data=True)
onnx.save_model(
    model,
    "merged-unified-model.onnx",
    save_as_external_data=False,
    size_threshold=0,  # Force everything into one file
)
```

Then upload `merged-unified-model.onnx` to MLflow. This is the simplest approach if your server has enough RAM during the merge operation.

**Option 2: Upload the entire export directory**

If merging consumes too much memory, keep the multi-part structure and upload everything:

```python
import mlflow

with mlflow.start_run():
    # Upload the whole export folder with both .onnx and .bin files
    mlflow.log_artifacts("onnx_export_dir", artifact_path="model")
```

When Ray Serve initializes and downloads the artifact, it will fetch both the graph and weight files together.

## Helpful Commands

```bash
# list Serve and RayService resources
kubectl get rayservice -n rationai-jobs-ns
kubectl get svc -n rationai-jobs-ns

# see all pods for a RayService
kubectl get pods -n rationai-jobs-ns -l ray.io/cluster=<release-name>
```

## Related Guides

- [Deployment Guide](deployment-guide.md)
- [Adding New Models](adding-models.md)
- [Configuration Reference](configuration-reference.md)
- [Optimization Guide](optimization-guide.md)
- [Architecture Overview](../architecture/overview.md)
