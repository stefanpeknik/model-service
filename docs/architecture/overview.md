# Architecture Overview

This page describes the Ray model-inference layer. For the workflow application,
durable scheduler, and provider boundaries, start at
[the documentation map](../README.md) and its package guides.

This section provides a structured overview of Model Service's architecture.

If you are new to the project, start here and then follow the links to the deeper pages.

## Overview

Problem solved: route HTTP inference requests to multiple Ray Serve applications while preserving clean service boundaries and independent scaling.

The diagram below summarizes the head-node control plane and worker-node execution plane used in this project.

## System Architecture

Model Service is built on Kubernetes + KubeRay + Ray Serve:

```text
+------------------------------------------------------------------+
|                           Head Node                              |
|                                                                  |
|    +--------------+                    +-------------------+     |
|    |  Controller  |<-------------------|    HTTP Proxy     |<---- Client Request
|    | (Autoscaler) |   Update Config    |     (Ingress)     |     |
|    +------+-------+                    +---------+---------+     |
|           |                                      |               |
+-----------+--------------------------------------+---------------+
            | Manage                               | Route
            v                                      v
+------------------------------------------------------------------+
|                         Worker Nodes                             |
|                                                                  |
|  +------------------------------------------------------------+  |
|  |                       Application 1                        |  |
|  |  +----------------------+        +----------------------+  |  |
|  |  |     Deployment A     |        |     Deployment B     |  |  |
|  |  | +--------+ +--------+|        | +--------+ +--------+|  |  |
|  |  | |Replica | |Replica ||        | |Replica | |Replica ||  |  |
|  |  | +--------+ +--------+|        | +--------+ +--------+|  |  |
|  |  +----------------------+        +----------------------+  |  |
|  +------------------------------------------------------------+  |
|                                                                  |
|  +------------------------------------------------------------+  |
|  |                       Application 2                        |  |
|  |               ...                                          |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
```

## Core Concepts & Hierarchy

The client's request flows through several layers of the system:

HTTP Proxy → Head Node → Worker Node → Application → Deployment → Replica.

Why this layering:

- Keep routing and control concerns in the head node.
- Keep model execution isolated in worker replicas.
- Allow multiple models (CPU/GPU) to coexist under one gateway.

The main components are:

1. **Ray Service (The Platform)**: The Kubernetes Custom Resource (CR) that defines the entire Ray cluster and the Serve application(s) running on top of it.
2. **Ray Cluster**: The physical set of Kubernetes pods, consisting of a **Head Node** and multiple **Worker Nodes**.
3. **Infrastructure Actors**:
   - **Controller**: Manages the control plane, API calls, and autoscaling (does not handle requests).
   - **HTTP Proxy**: Ingress point that routes requests to applications.
4. **Serve Application (The Service Boundary)**: A standalone version of your code, including all its deployments and logic. Defined by an import path (e.g., `models.binary_classifier:app`).
5. **Serve Deployment (The Functional Unit)**: A managed group of replicas. It defines scaling rules (`num_replicas`, `num_cpus`) and versioning.
6. **Replica (The Execution Unit)**: A single Ray actor process running the deployment code inside a Worker Node.

### Serve application vs Serve deployment

- **Application**: deployable service boundary (routing, code entrypoint, runtime env).
- **Deployment**: scaling unit (replicas), concurrency/queue limits, and resource options.

### Internal Mechanisms

For detailed information on how batching works, including the configuration API and internal buffering mechanisms, see [Batching](batching.md).

For request lifecycle and queueing details, see [Request Lifecycle](request-lifecycle.md) and [Queues and Backpressure](queues-and-backpressure.md).

## Scaling Architecture

### Horizontal Scaling (Replicas)

Models scale horizontally by adding/removing replicas.

**Autoscaling Triggers:**

- `target_ongoing_requests`: Target requests per replica
- Scale up when: requests > (replicas × target)
- Scale down when: requests < (replicas × target)

### Vertical Scaling (Workers)

Ray cluster scales by adding/removing worker pods:

```yaml
workerGroupSpecs:
  - groupName: cpu-workers
    minReplicas: 0
    maxReplicas: 4
```

### Resource Sizing (Pods vs Replicas)

It is important to distinguish between **Kubernetes Resources** (Pods) and **Ray Resources** (Replicas).

- **Replica Sizing (`ray_actor_options`)**: Defines how much logical resource one model copy needs (e.g., `num_cpus: 1`).
- **Pod Sizing (`resources.limits`)**: Defines how big the physical container is.

**Rule of Thumb**: Ensure your Pods are large enough to fit at least one (or N) replicas plus overhead (Python runtime, Object Store).
i.e., `Pod CPU >= Replicas × num_cpus + Overhead`.

## Autoscaling Architecture

The Ray Serve Autoscaler runs inside the **Controller** actor and manages the number of replicas dynamically.

1. **Metrics Collection**: Replicas and DeploymentHandle push metrics (queue size, active queries) to the Controller.
2. **Decision Making**: The Autoscaler periodically checks these metrics against targets (like `target_ongoing_requests`).
3. **Scaling Action**: The Controller adds or removes Replica actors to meet demand.

## Fault Tolerance

Ray Serve is designed to be resilient to failures:

- **Replica Failure**: If a Replica actor crashes, the Controller detects it and starts a new one to replace it. Request routing automatically updates.
- **Proxy Failure**: If the Proxy actor fails, the Controller restarts it.
- **Controller Failure**: If the Controller itself fails, Ray (via GCS) restarts it. Autoscaling pauses during downtime but resumes upon recovery.
- **Node Failure**: KubeRay (managing the cluster) detects node failures and provisions new pods. Ray Serve then eventually schedules actors on the new nodes.

## Design Principles

1. **Declarative Configuration**: Infrastructure defined in YAML, assembled with Helm.
2. **Separation of Concerns**: Model Code (Python), Infrastructure (K8s), Configuration (User Config).
3. **Elastic Scaling**: Scale to zero when idle, scale up on demand.
4. **Developer Experience**: Simple model implementation, easy local testing.

## Metrics & Debugging

Common commands:

```bash
kubectl get pods -n rationai-jobs-ns
kubectl top pods -n rationai-jobs-ns
kubectl logs -n rationai-jobs-ns <pod-name>
kubectl describe rayservice <release-name> -n rationai-jobs-ns
```

Ray can export Prometheus metrics (when metrics collection/export is enabled):

- Request latency
- Request throughput
- Replica count
- Resource usage

## Next Steps

- [Request lifecycle](request-lifecycle.md)
- [Deployment guide](../guides/deployment-guide.md)
- [Configuration reference](../guides/configuration-reference.md)
- [Adding new models](../guides/adding-models.md)
