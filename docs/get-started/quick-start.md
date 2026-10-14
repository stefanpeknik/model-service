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
## Step 2: Add your model code
If you want to deploy your own model, start by adding the model code and Ray Serve application entry point. Follow the [Adding New Models](../guides/adding-models.md) guide for a step-by-step walkthrough of this process.

## Step 3: Review the Configuration

In Model Service, configurations are managed using **Helm**. The environment configuration is split into values and applications inside the `helm/rayservice/` directory.

Models are configured by YAML files in the `helm/rayservice/models/` folder.

Let's look at a sample model file (`helm/rayservice/models/prostate-classifier-1.yaml`):

```yaml
inference:
  contract: spatial-scores
serve:
  name: prostate-classifier-1
  import_path: models.binary_classifier:app
  route_prefix: /prostate-classifier-1
  # ...
```

**Let's break down the application config:**

- `name`: Logical app name (visible in the Ray dashboard and logs).
- `import_path`: Python entrypoint (`module.path:variable`).
- `route_prefix`: HTTP path under the Serve gateway.
- `inference.contract`: how builders call the model; `heatmap` accepts `spatial-scores`.

For your first deployment, we will use the existing configuration without changes.

---

## Step 4: Deploy the service

To deploy the service, run Helm:

```bash
helm upgrade --install <release-name> helm/rayservice -n rationai-jobs-ns
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
