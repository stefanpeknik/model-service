# Model Service Documentation

For builders, durable execution and the model catalog, start at
[the architecture](architecture/core.md) and [the documentation map](README.md).
The guides below cover deploying and writing models.

## Overview

Model Service is a Helm based deployment framework for serving ML models on Kubernetes with Ray Serve.

Problem solved:
turn Python model code into stable HTTP endpoints that can batch requests, scale replicas, and run on CPU or GPU workers.

Use this documentation when:

- you want to deploy a model as an API endpoint,
- you need to tune throughput and latency,
- or you operate multi-model workloads on one Ray cluster.

## What You Get

- Ray Serve applications managed by a Helm chart in `deploy/helm/rayservice/`.
- Per-model configuration in `deploy/helm/rayservice/applications/`.
- Default worker profiles in `deploy/helm/rayservice/workers/`.
- Operational guides for deployment, scaling, and troubleshooting.

## Two Ways to Use the Service

Depending on your goal, you will either run a pre-built model or deploy a custom model:

1. **Run an existing model**
   If the model you need is already in [Available Models](available-models.md), you just install the Helm chart.
   _Action:_ Go to [Quick Start](get-started/quick-start.md).

2. **Deploy a new model**
   If you're bringing a new ML architecture, follow the [Deployment Guide](guides/deployment-guide.md) which will guide you through the entire flow: Python implementation, Helm config, and deployment.

## Start Here

### I want to run an existing model

If the model is already implemented and you just need to spin it up on the cluster.

1. [Quick Start](get-started/quick-start.md)
2. [Deployment Guide](guides/deployment-guide.md) (steps 2+: Helm config and deploy)
3. [Troubleshooting](guides/troubleshooting.md)

### I want to deploy a new model

If you are bringing a new ML architecture and need to write a custom Ray Serve application.

Start with the [Deployment Guide](guides/deployment-guide.md):

1. **Step 1**: Prepare Python Entrypoint → this will point you to [Adding New Models](guides/adding-models.md)
2. **Step 2–6**: Return to [Deployment Guide](guides/deployment-guide.md) for Helm config and deployment
3. [Optimization Guide](guides/optimization-guide.md) (optional, for tuning performance)
4. [Troubleshooting](guides/troubleshooting.md) (if anything goes wrong)

## Documentation Map

### Getting Started

- [Quick Start](get-started/quick-start.md): first deployment using a dedicated test release name.

### Guides

- [Adding New Models](guides/adding-models.md): implement model code and bind routes.
- [Deployment Guide](guides/deployment-guide.md): safe deployment workflow and production checklist.
- [Configuration Reference](guides/configuration-reference.md): source of truth for Helm and Ray Serve settings.
- [Optimization Guide](guides/optimization-guide.md): TensorRT, precision, batch tuning, and session options.
- [Troubleshooting](guides/troubleshooting.md): diagnostics for deployment and runtime failures.

### Models

- [Available Models](available-models.md): list of pre-configured models with endpoints and SDK patterns.

### Architecture

- [Overview](architecture/overview.md): system components and scaling model.
- [Request Lifecycle](architecture/request-lifecycle.md): end-to-end request flow.
- [Queues and Backpressure](architecture/queues-and-backpressure.md): overload behavior and queue tuning.
- [Batching](architecture/batching.md): how request coalescing works per replica.

## Important Notes

- Use a dedicated test release name (for example `rayservice-model-<test>`) while experimenting.
- Keep default worker profiles unless you need specific hardware or scheduling behavior.
- Tune application/deployment settings first, worker templates second.

## Glossary

- RayService: KubeRay custom resource managing a Ray cluster plus Serve applications.
- Deployment (Ray Serve): scalable unit running one part of model code.
- Replica: one running instance of a deployment.
- Worker group: pool of Ray worker pods with its own resource profile.

## Related Links

- [GitHub repository](https://github.com/RationAI/model-service)
- [KubeRay installation guide](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/kuberay-operator-installation.html)
- [Ray Serve on Kubernetes](https://docs.ray.io/en/latest/serve/production-guide/kubernetes.html)
