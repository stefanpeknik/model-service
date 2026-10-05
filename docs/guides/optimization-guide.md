# Optimization Guide

This guide collects runtime optimization options for ONNX-based models in Model Service. Use it after your model is already exported to ONNX, uploaded to MLflow, and wired into the Python entrypoint.

## When to Use This Guide

Use these settings when you want to improve inference latency, throughput, or GPU utilization without changing the model architecture.

Typical cases:

- TensorRT acceleration on NVIDIA GPUs.
- CUDA or CPU execution provider selection.
- Mixed precision such as `float16`.
- Batch size and queue tuning.
- External weight or model loading behavior during startup.

## TensorRT Execution Provider

TensorRT is usually the first optimization to try on NVIDIA hardware. It can reduce latency and improve throughput by building optimized inference kernels for the exported ONNX graph.

### How It Works

1. **Model Compilation**: On first run, TensorRT analyzes your ONNX graph and builds an optimized engine for your specific GPU model and batch size.
2. **Engine Caching**: The built engine can be cached to disk, so subsequent restarts skip the compilation step.
3. **Inference**: TensorRT uses highly tuned kernels for inference, often 2–5x faster than generic ONNX Runtime.

### Setup in `reconfigure()`

Add TensorRT options to the session provider list in your model's `reconfigure` method:

```python
def reconfigure(self, config: Config) -> None:
  import onnxruntime as ort
  
  # Create cache directory for TensorRT engines
  cache_path = config["trt_cache_path"]
  os.makedirs(cache_path, exist_ok=True)

  # Define batch profile shapes for TensorRT optimization
  min_shape = f"input:1x3x{self.tile_size}x{self.tile_size}"
  opt_shape = f"input:{config['max_batch_size']}x3x{self.tile_size}x{self.tile_size}"
  max_shape = f"input:{config['max_batch_size']}x3x{self.tile_size}x{self.tile_size}"

  # TensorRT options
  trt_options = {
    "device_id": 0,                           # GPU device index
    "trt_fp16_enable": True,                  # Use float16 for faster inference
    "trt_engine_cache_enable": True,          # Cache compiled engines to disk
    "trt_engine_cache_path": cache_path,      # Where to store cached engines
    "trt_timing_cache_enable": True,          # Cache kernel timing info for faster rebuilds
    "trt_max_workspace_size": 8 * 1024**3,    # 8GB workspace for kernel search (default 1GB is too small)
    "trt_builder_optimization_level": 1,      # 1=fast build, 5=slow but highly optimized
    "trt_profile_min_shapes": min_shape,      # Minimum input shape for batching
    "trt_profile_max_shapes": max_shape,      # Maximum input shape for batching
    "trt_profile_opt_shapes": opt_shape,      # Expected typical shape for optimization
  }

  # Create ONNX Runtime session with TensorRT
  self.session = ort.InferenceSession(
    provider_path,  # Path to ONNX model from MLflow
    providers=[
      ("TensorrtExecutionProvider", trt_options),  # Try TensorRT first
      "CUDAExecutionProvider",                      # Fallback to CUDA
      "CPUExecutionProvider",                       # Final fallback to CPU
    ],
  )
```

### Key Parameters

| Parameter | Purpose | Default | When to Adjust |
|-----------|---------|---------|-----------------|
| `trt_fp16_enable` | Use float16 precision | False | Set to True for faster inference on Tensor Cores (if model tolerates it) |
| `trt_engine_cache_enable` | Save compiled engines | False | Set to True to avoid rebuilding on container restart |
| `trt_engine_cache_path` | Directory for cached engines | N/A | Point to a persistent volume (e.g., ConfigMap or PVC in Helm) |
| `trt_max_workspace_size` | GPU memory for kernel search | 1 GB | Increase to 4–8 GB for high-res models to find better kernels |
| `trt_builder_optimization_level` | Build speed vs quality | 1 | Use 1 for fast development, 3–5 for production |
| `trt_profile_*_shapes` | Batch size range | required | Must match your `max_batch_size` from config |

### Where to Put the Cache Path in Helm

Store the cache path in a shared location that persists across pod restarts.

**Option 1: Use an emptyDir volume (data lost on pod restart)**

In your application YAML in `helm/rayservice/applications/`, set the environment variable:

```yaml
deployments:
  - name: BinaryClassifier
    ray_actor_options:
      runtime_env:
        env_vars:
          TRT_CACHE_PATH: /data/trt-cache
```

Then add volume mounting to your worker group in `helm/rayservice/values.yaml`:

```yaml
rayClusterConfig:
  workerGroupSpecs:
    - groupName: gpu-group
      rayStartParams:
        num-gpus: 1
      template:
        spec:
          containers:
            - name: ray-worker
              volumeMounts:
                - name: trt-cache
                  mountPath: /data/trt-cache
          volumes:
            - name: trt-cache
              emptyDir: {}
```

**Option 2: Use a PersistentVolumeClaim (data persists across restarts)**

For production, use a PVC to ensure TensorRT engines persist:

```yaml
# In worker group template
volumeMounts:
  - name: trt-cache
    mountPath: /data/trt-cache
volumes:
  - name: trt-cache
    persistentVolumeClaim:
      claimName: trt-cache-pvc
```

Create the PVC separately (or reference an existing one):

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: trt-cache-pvc
  namespace: rationai-jobs-ns
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
```

Once a TensorRT engine is built and cached, subsequent pod restarts will reuse it, avoiding the 1–5 minute compilation overhead.

### First Run Overhead

The first deployment or after cache cleanup, TensorRT will take **1–5 minutes** to compile the engine, depending on model size and `trt_builder_optimization_level`. Plan for this during initial tests.

### Fallback Order

Typical provider order:

```python
providers=[
  ("TensorrtExecutionProvider", trt_options),
  "CUDAExecutionProvider",
  "CPUExecutionProvider",
]
```

Notes:

- Put TensorRT first so ONNX Runtime prefers it when available.
- Keep CUDA and CPU providers as fallbacks.
- If TensorRT is not available (e.g., on CPU-only nodes), ONNX Runtime automatically falls back to CUDA, then CPU.

## Precision and Dtype

Many models can run faster with reduced precision, especially on GPUs.

### Where Precision is Decided

Precision is determined in three places, in this order:

1. **ONNX Export** (highest priority): If you export the model as `float16` or quantized, the artifact inherits that precision.
2. **Runtime Execution Provider** (middle): TensorRT's `trt_fp16_enable: True` tells TensorRT to use float16 kernels for inference.
3. **Input Data** (lowest priority): The dtype of the input tensor to `session.run()`.

### Common Options

- **float32** (default): Maximum compatibility, typically required for accuracy-sensitive models.
- **float16**: ~2x speedup on Tensor Cores, acceptable for most image models, often imperceptible loss of accuracy.
- **int8**: Quantized inference, requires calibration, best 3–4x speedup, but model must be trained/exported for it.
- **Mixed precision**: Export key layers as float16, others as float32. Requires custom export logic.

### How to Use

For most models, enable float16 in TensorRT:

```python
trt_options = {
  "trt_fp16_enable": True,
  # ... other options
}
```

If the model was exported as float16 in ONNX:

```python
import onnx
model = onnx.load("model_fp16.onnx")
# model graph already uses float16 initializers and ops
```

If you need float32 inference despite the export, ensure input is cast:

```python
@serve.batch
async def predict(self, images: list[NDArray]) -> list[float]:
  batch = np.stack(images, axis=0, dtype=np.float32)  # Force float32
  outputs = self.session.run([self.output_name], {self.input_name: batch})
  return outputs[0].flatten().tolist()
```

### Testing Precision

Start with float32 to confirm the model runs correctly, then enable float16 and compare outputs:

1. Benchmark inference time with float32.
2. Enable `trt_fp16_enable: True` and re-benchmark.
3. Measure output differences (should be <1% for most models).
4. If differences are acceptable, keep float16 enabled.

## Batch and Queue Tuning

Ray Serve batching is often the simplest throughput optimization. When many HTTP requests arrive, Ray Serve collects them into one batch and passes them together to the `@serve.batch` decorated method.

### Where to Set It

In the model's `reconfigure()` method, right after creating the ONNX Runtime session:

```python
def reconfigure(self, config: Config) -> None:
  # ... load session ...
  
  self.predict.set_max_batch_size(config["max_batch_size"])
  self.predict.set_batch_wait_timeout_s(config["batch_wait_timeout_s"])
```

These values come from the Helm config:

```yaml
user_config:
  max_batch_size: 16
  batch_wait_timeout_s: 0.1
```

### Parameters

| Parameter | Meaning | Example | Trade-off |
|-----------|---------|---------|-----------|
| `max_batch_size` | Maximum requests to batch together before forcing inference | 16 | Larger = higher throughput, higher latency |
| `batch_wait_timeout_s` | How long to wait for more requests before inferencing a smaller batch | 0.1 | Longer = better batching, higher latency |

### Tuning Strategy

**For low-latency workloads:**
- `max_batch_size: 2–4`
- `batch_wait_timeout_s: 0.01` (10ms)
- Trade: Each inference runs on fewer samples, but requests are processed faster.

**For high-throughput workloads:**
- `max_batch_size: 32–64`
- `batch_wait_timeout_s: 0.2–0.5` (200–500ms)
- Trade: Larger batches maximize GPU utilization and throughput, but single requests may wait longer.

**Rule of thumb:**
- Monitor GPU utilization. If <50% busy, increase batch size or timeout.
- If latency is >200ms unacceptable, reduce batch size and timeout.

### Memory and Batch Size

Memory required ≈ (model size) + (batch size × input size × 2–3 for computations).

Example: A 500MB model + batch 16 of 512×512×3 uint8 images:
- Model: 500 MB
- Input batch: 16 × 512 × 512 × 3 bytes ≈ 12 MB
- Activations and intermediates: ~50–100 MB
- **Total: ~600 MB**

Allocate accordingly in Helm `ray_actor_options.memory`.

## Startup and Artifact Loading

If model startup is slow, check whether the artifact is large or split into multiple files.

For large exports, see [Troubleshooting: Large ONNX Model Export](troubleshooting.md#large-onnx-model-export-2gb).

If the model is loaded from MLflow, keep the artifact structure simple and ensure the runtime can access the full set of files during replica initialization.

## Session Options

ONNX Runtime session options control graph optimization and threading behavior.

### Key Options

In `reconfigure()`, configure session options before creating the session:

```python
import onnxruntime as ort

sess_options = ort.SessionOptions()

# Threading
sess_options.intra_op_num_threads = 2      # Threads per operation (e.g., matrix multiply)
sess_options.inter_op_num_threads = 1      # Threads for parallelizing different ops
                                            # Set to 1 when using external batching via @serve.batch

# Graph optimization
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Execution mode
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # Sequential = less overhead for small batches

self.session = ort.InferenceSession(
  provider_path,
  sess_options=sess_options,
  providers=[...],
)
```

### Threading Strategy

When using `@serve.batch` (external batching by Ray):

- `intra_op_num_threads = 2–4`: Allow individual ops to use multiple threads.
- `inter_op_num_threads = 1`: Avoid parallelizing different ops (Ray handles parallelism via replicas).

When using in-graph batching or single-request mode:

- `intra_op_num_threads = cpu_count / 2`: Let individual ops use more threads.
- `inter_op_num_threads = 1–2`: Limited inter-op parallelism is usually better than full.

### Graph Optimization

Always use `ORT_ENABLE_ALL` to enable:

- Constant folding (pre-compute static values).
- Node fusion (combine multiple ops into one kernel).
- Layout optimization (reorder data for cache efficiency).

This has no runtime cost and can significantly speed up inference.

## Related Guides

- [Adding New Models](adding-models.md)
- [Deployment Guide](deployment-guide.md)
- [Configuration Reference](configuration-reference.md)
- [Troubleshooting](troubleshooting.md)
- [Architecture Overview](../architecture/overview.md)
