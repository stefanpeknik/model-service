# Batching (How It Works Under the Hood)

Batching in Ray Serve is a **replica-local request coalescing** mechanism.

It improves throughput when your model can process multiple inputs more efficiently together (common for GPU inference).

## Where batching happens

Batching happens **inside each replica process**.

Requests only become eligible for batching after they:

1. enter through the proxy and handle queueing/backpressure, and
2. get routed to a specific replica

See also: **[Request lifecycle](request-lifecycle.md)**.

## What you configure

In plain Ray Serve code, batching is enabled by decorating an **async** method
with `@serve.batch` and its `max_batch_size` and `batch_wait_timeout_s`.
Served models don't need that code: the generic host batches with its own
batcher (`src/model_service/served_model/batching.py`), which behaves the
same way and runs off Serve too, so the host can be tested without Ray. A
served-model file sets it:

- `batching.max_size`: the most tiles in one batch (`max_batch_size`);
- `batching.wait_s`: how long the first tile waits for more before a smaller
  batch leaves (`batch_wait_timeout_s`).

The runtime returns one output per tile, in order.

## What Serve actually does internally

Conceptually, each replica maintains an internal structure like:

- an in-memory buffer of pending calls
- a background “flush” loop that decides when to execute a batch
- per-request futures/promises that get completed when the batch finishes

### 1. Collection phase (buffering)

Incoming requests that hit the batched method are appended to a replica-local buffer.

Each buffered entry stores:

- the request arguments (or decoded payload)
- a future representing that request’s eventual response

### 2. Flush conditions (size or time)

The buffer is flushed when either condition becomes true:

- **Size trigger**: buffer length reaches `max_batch_size`
- **Time trigger**: `batch_wait_timeout_s` elapses since the **first** item currently in the buffer

This is why batching can increase latency at low QPS: a request may wait up to `batch_wait_timeout_s` for more arrivals.

### 3. Execution phase (single call)

Serve invokes your batched handler **once** with a list of inputs.

This is where you typically vectorize:

- stack/concat tensors
- run one forward pass
- split/scatter outputs back

### 4. Scatter phase (complete futures)

When the batched handler returns a list of outputs, Serve resolves the stored futures in order.

Each original HTTP request then completes independently with its corresponding output.

## Configuration & Tuning

For a deep dive into how batching interacts with concurrency limits (specifically why `max_ongoing_requests` must be larger than `max_batch_size`), see **[Queues and backpressure](queues-and-backpressure.md)**.

Quick tips:

- Increase `batching.max_size` if the model benefits from larger batches and you have headroom.
- Increase `batching.wait_s` to favor fuller batches; decrease it to favor latency.

## Next

- Request flow including queue points: [Request lifecycle](request-lifecycle.md)
- Queueing and rejection controls: [Queues and backpressure](queues-and-backpressure.md)
- What each block of a served-model file sets: [How the chart deploys the catalog](deployment.md)
