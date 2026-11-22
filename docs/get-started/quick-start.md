# Quick Start

In this tutorial, you will deploy your first model to a Kubernetes cluster.

You will learn:

- How to get the project code.
- How to review and apply the Helm configuration.
- How to verify the deployment status.
- How to send requests to your running model.

## Prerequisites

Before you begin, ensure you have the following:

- Access to a Kubernetes cluster with the **KubeRay operator** installed. If you do not have KubeRay installed, follow the [Installation Guide](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/kuberay-operator-installation.html).
- The `kubectl` command-line tool configured to communicate with your cluster.
- Basic familiarity with Kubernetes concepts such as pods and namespaces.

---

## Step 1: Get the code

Open your terminal and clone the repository:

```bash
git clone https://github.com/RationAI/model-service.git
cd model-service
```

---
## Step 2: Add your model
A model the existing components can run (an ONNX graph, a timm encoder) needs only a YAML file in `catalog/served-models/`; see the [Configuration Reference](../guides/configuration-reference.md#2-served-model-files). For your first deployment, use the models already there.

## Step 3: Review the Configuration

In Model Service, configurations are managed using **Helm**. The chart and its values files are in `deploy/helm/rayservice/`; the chart reads the models from the repo's `catalog/`.

Each model is one YAML file in `catalog/served-models/`, named by its model ID.

Let's look at a sample model file (`catalog/served-models/prostate-classifier-1.yaml`):

```yaml
weights: {use: mlflow, uri: "mlflow-artifacts:/65/…/prostate_model_norm.onnx"}
runtime: {use: onnx, tensorrt: true, workspace_gib: 8, optimization_level: 3, threads: 4}
input: {tile_size: 512, mpp: 0.468}
output: {kind: scores, channels: 1, size: 1, range: [0, 1]}
batching: {max_size: 256, wait_s: 0.05}
resources: {gpus: 1, cpus: 4, memory_gib: 12}
```

**Let's break down the file:**

- `weights`: where the trained weights come from.
- `runtime`: what runs them, here ONNX Runtime with TensorRT.
- `input`, `output`: the tile the model takes (size in pixels, resolution in µm per pixel) and what it returns, here one score per tile.
- `batching`, `resources`: how many tiles one call takes, and what one replica reserves.

The chart turns the file into a Ray Serve application named `prostate-classifier-1` under the route `/prostate-classifier-1`. The values file's `servedModels` picks which files deploy (`["*"]` for all).

For your first deployment, we will use the existing configuration without changes.

---

## Step 4: Deploy the service

To deploy the service, run Helm:

```bash
helm upgrade --install <release-name> deploy/helm/rayservice -n rationai-jobs-ns
```

Use a dedicated test release name (for example `rayservice-model-my-model`) so test actions do not affect the main deployment.

This command automates the deployment process by compiling your Helm templates and applying the final manifest to the Kubernetes cluster.

Code runs from the image, so a code change needs a new image tag in the values file.

**Tip for avoiding cache issues:** Ray caches the downloaded `working_dir` based on the URL string. If you push new code to the same branch/zip URL, Ray will use the old cached version. A great way to bypass this and force a refresh is to add a query parameter to the `working_dir` URL in your config, like `?v=1`, `?v=2`, etc. You can just do this locally before deploying; it doesn't even need to be pushed to the remote repo.

---

## Step 5: Monitor the deployment

Deploying models takes time as the cluster downloads images and starts worker pods.

Check the overall status of your RayService:

```bash
kubectl get rayservice <release-name> -n rationai-jobs-ns
```

Check the status of the individual pods:

```bash
kubectl get pods -n rationai-jobs-ns
```

If the pods fail to start, you can inspect the details for troubleshooting:

```bash
kubectl describe rayservice <release-name> -n rationai-jobs-ns
```

### Note: Using the Ray Dashboard

Ray provides a dashboard for visual monitoring. To access it, forward the port to your local machine:

```bash
kubectl port-forward -n rationai-jobs-ns svc/<release-name>-head-svc 8265:8265
```

Open a web browser and navigate to `http://127.0.0.1:8265`. Your models are ready when their Serve applications display a **RUNNING** status.

---

## Step 6: Send a request

To communicate with the model from your local machine, forward the Serve port:

```bash
kubectl port-forward -n rationai-jobs-ns svc/<release-name>-serve-svc 8000:8000
```

You can now send HTTP requests to `http://localhost:8000/prostate-classifier-1/`.

---

## Step 7: Clean up

When you are finished, delete the deployment to free up cluster resources:

```bash
helm uninstall <release-name> -n rationai-jobs-ns
```

This removes only your selected `<release-name>` test deployment.

---

## Related Guides

- To deploy your own custom Python model, see [Adding New Models](../guides/adding-models.md).
- To configure scaling or memory settings, read the [Configuration Reference](../guides/configuration-reference.md).
- For a comprehensive walkthrough of a production deployment, see the [Deployment Guide](../guides/deployment-guide.md).
- To understand the internal architecture, explore the [Architecture Overview](../architecture/overview.md).
