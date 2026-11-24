# Request Lifecycle in Detail

This document traces the path of a single inference request through the Model Service stack, from the external HTTP client down to the Ray Core task execution.

It also highlights **where requests queue** and which settings control queueing vs rejection.

## High-Level Flow

Overview: one ingress can expose multiple model routes, each backed by its own Serve application and deployment policy.

```text
+--------------------------------------------------------------------+
|                           External Client                           |
|                                                                    |
| HTTP Request                                                       |
+-------------------------------+------------------------------------+
                                |
                                v
+--------------------------------------------------------------------+
|                       K8s Service / Ingress                        |
+-------------------------------+------------------------------------+
                                |
                                v
+--------------------------------------------------------------------+
|                       Head / Worker Nodes                          |
|                                                                    |
|  +---------------------------+                                     |
|  | HTTP Proxy Actor          |                                     |
|  | (ServeHTTPProxy)          |                                     |
|  +-------------+-------------+                                     |
|                | create DeploymentHandle                           |
|                v                                                   |
|  +---------------------------+                                     |
|  | Deployment Handle         |                                     |
|  | (client-side queue)       |                                     |
|  +-------------+-------------+                                     |
|                | enqueue / backpressure                            |
|                v                                                   |
|  +---------------------------+                                     |
|  | Serve Router              |                                     |
|  | (replica selection)       |                                     |
|  +-------------+-------------+                                     |
|                | PushTask RPC                                      |
|                v                                                   |
|  +---------------------------+                                     |
|  | Replica Actor             |                                     |
|  | (deployment instance)     |                                     |
|  +-------------+-------------+                                     |
|                | execute                                           |
|                v                                                   |
|  +---------------------------+                                     |
|  | Ray Worker Process        |                                     |
|  +-------------+-------------+                                     |
|                v                                                   |
|  +---------------------------+                                     |
|  | User Model Code           |                                     |
|  | (inference / logic)       |                                     |
|  +-------------+-------------+                                     |
|                | result                                            |
|                v                                                   |
|  +---------------------------+                                     |
|  | Plasma Object Store       |  <- large payloads (optional)       |
|  +-------------+-------------+                                     |
|                | ObjectRef / inline                                |
|                v                                                   |
|  +---------------------------+                                     |
|  | HTTP Proxy Actor          |                                     |
|  +-------------+-------------+                                     |
|                | HTTP Response                                     |
+----------------+---------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------------+
|                           External Client                           |
|                         200 OK Response                             |
+--------------------------------------------------------------------+
```

## Step-by-Step Breakdown

### 1. Ingress (HTTP Proxy)

**Component**: `ServeHTTPProxy` actor (running on Head or Worker nodes).

1.  **Receive**: The request hits the Uvicorn server running inside the Proxy actor.
2.  **Route Matching**: The proxy inspects the URL path to match it against active **Applications** and their **Ingress Deployments**.
3.  **Handle Creation**: The proxy uses a `DeploymentHandle` to forward the request. It does **not** send the request directly to a replica yet.

### 2. Queueing & Backpressure (Deployment Handle)

**Component**: `DeploymentHandle` (client-side in the Proxy).

The request enters a **Handle Queue** managed by the caller (the Proxy).

- **Assignment**: The handle checks for available slots in the target Deployment.
- **Backpressure**: If replicas are saturated (`max_ongoing_requests`), the request stays in this queue instead of being pushed to a replica.
- **Rejection**: If the handle queue grows beyond `max_queued_requests`, the request is rejected with an overload-style error (client-visible backpressure).

**Where this queue lives**: inside the process that is making the call (here: the HTTP Proxy). It is not a replica-local queue.

### 3. Replica Assignment (Ray Core)

**Component**: `ServeRouter` & `Ray Core`.

When a slot is available:

1.  **Routing**: The router selects a specific Replica actor ID based on the policy (e.g., `PowerOfTwoChoices`).
2.  **RPC**: The request is serialized and sent via Ray's internal gRPC protocol to the selected actor.

> **Under the Hood: Ray Task Lifecycle**
>
> - **Submission**: The router behaves like a Ray Core driver submitting a task.
> - **Worker Lease**: Ray guarantees the actor exists. If the actor had crashed, the Ray Controller would have already requested a new worker lease from the **Raylet** to restart it.
> - **PushTask**: The `PushTask` RPC carries the request data.

### 4. Execution (Worker & Replica)

**Component**: `RayWorker` process.

1.  **Receive**: The Worker process hosting the Replica actor receives the message.
2.  **Deserialization**:
    - **Small Data**: Unpickled directly from the message.
    - **Large Data**: If the request payload is large, it may be retrieved from the **Plasma Object Store** (shared memory).
3.  **Asyncio Loop**: The request enters the actor's entrypoint (usually `__call__`).
4.  **Replica Concurrency Limit**: The replica will not run more than `max_ongoing_requests` concurrently. Requests beyond that should not be dispatched to this replica; instead they remain queued at the caller-side handle.
5.  **Batching**: The request waits in the replica's batching buffer until `batching.max_size` tiles are there or `batching.wait_s` has passed (see [Batching](batching.md)).
6.  **Inference**: The model's runtime runs on the batch, between its `pre:` and `post:` steps.

### 5. Response & Return

**Component**: Shared Memory & Network.

1.  **Completion**: The function returns a result.
2.  **Storage**:
    - **Small Result**: Sent back directly in the RPC response.
    - **Large Result**: Stored in the local Plasma Store; only an `ObjectRef` is returned.
3.  **Forwarding**: The HTTP Proxy waits for the result (resolving the `ObjectRef` if necessary) and writes the HTTP response body.
4.  **Client**: The client receives the `200 OK`.

## Where queues are handled (and where requests get rejected)

Ray Serve has multiple queue-like stages. They serve different purposes and are controlled by different knobs.

For deep-dive explanation and tuning advice, see **[Queues and Backpressure](queues-and-backpressure.md)**.

### 1. Proxy-side “handle queue” (caller-side)

When an HTTP request hits Ray Serve, the proxy forwards it through a `DeploymentHandle`.
That handle maintains a **caller-side queue** of requests waiting to be assigned to a replica.

This is where `max_queued_requests` applies.

- If replicas are busy (because of per-replica concurrency limits), the request waits here.
- If the queue grows beyond `max_queued_requests`, the request is rejected (client-visible backpressure).

### 2. Routing / replica selection

Once a request can be dispatched, Ray Serve selects a replica.

This stage is not intended to be a long-term queue - it is primarily where the system decides _which_ replica gets the request next.

### 3. Replica concurrency slots (“ongoing requests”)

Each replica enforces a cap on concurrent in-flight work via `max_ongoing_requests`.

- If a replica already has `max_ongoing_requests` in progress, new work should not be scheduled onto it.
- “Ongoing” includes requests that are actively executing _or_ are awaiting completion (e.g., waiting for I/O or for a batch to flush).

### 4. Replica-local batching buffer (optional)

If you use `@serve.batch`, requests assigned to the replica can enter a **batching buffer** inside the replica.

This buffer is flushed when either:

- it reaches `max_batch_size`, or
- `batch_wait_timeout_s` elapses since the first buffered request

This buffer is not controlled by `max_queued_requests` (that limit is caller-side).

**[Queues and backpressure](queues-and-backpressure.md)** explains specifically how `max_ongoing_requests` and `max_queued_requests` interact.
