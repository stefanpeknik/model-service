# Adding New Models

## A model that fits an existing implementation: YAML only

Most new models are another trained artifact for an implementation that already
exists in `src/model_service/model_implementations/`: another binary classifier ONNX, another segmentation model,
another tile encoder. These need no Python and no image rebuild.

1. Copy the closest file in `deploy/helm/rayservice/models/` (for example
   `prostate-classifier-1.yaml`) to `deploy/helm/rayservice/models/<model-id>.yaml`.
2. Set `serve.name` and `serve.route_prefix: /<model-id>`, the artifact under
   `user_config.model`, and the model's geometry in `user_config`
   (`tile_size`, `output_tile_size`, `n_channels`, `mpp` for a classifier or
   segmentation model).
3. Keep `inference.contract` (`spatial-scores` for classifiers and segmentation,
   `tile-embedding` for encoders). An encoder also declares `inference.mpp` and
   `inference.summaries`, because its implementation does not read them.
4. Add `<model-id>` to `models:` in the values file and run `helm upgrade`.
5. Check `GET /v1/models`: the model is listed with the builders that can use it,
   for example `heatmap`. Run it with
   `client.builders.run("heatmap", BuilderRequest(model_id="<model-id>", ...))`.

The API refuses to start if a model file is inconsistent (wrong
`route_prefix`, missing geometry, a parameter declared twice), so a mistake shows
up at `helm upgrade` rather than at the first request.

Declare truthful values: the contract says how to call the model, not whether its
predictions mean what you expect. `mpp` in particular must be the resolution the
model was trained at.

## Iterating on a model

While the model is still changing, the Python implementation is written once
and each new training run only changes YAML:

1. Point `user_config.model.artifact_uri` in
   `deploy/helm/rayservice/models/<id>.yaml` at the new MLflow artifact.
2. Update `tile_size`, `mpp` and the other geometry fields if the new run
   changed them; the API checks them against the running model.
3. `helm upgrade`. Ray Serve calls `reconfigure`, which loads the new artifact. The TensorRT engine is cached per
   checkpoint, so the first start after a new checkpoint builds a new one.

A change to the Python implementation needs a new image:
`deploy/images/Dockerfile.gpu` copies `src/model_service/` into it. Tag it,
push it and set the image in the values file.

To try the model, call it directly for a quick answer, or schedule a builder
over many slides and let it queue:

- `client.builders.run("heatmap", BuilderRequest(model_id="<id>", ...))` waits
  for the result.
- `client.builders.schedule(...)` stores the execution in PostgreSQL and
  returns at once. It waits in the queue until the model has a free replica
  (GPU workers scale from zero), survives API restarts, and is polled with
  `client.executions.get(id)` or cancelled. Scheduled work has no start-time
  option; it runs as soon as capacity allows.
  For many slides, call `schedule` once per slide; there is no batch call.
  Each model runs at most its configured replicas, so queued work never takes
  more GPUs than the model file allows.
- Raw per-tile inference is not part of the API: post one LZ4-compressed
  `tile_size`×`tile_size`×3 uint8 tile to the model's Ray Serve ingress at
  `/<id>/` (`scripts/warm_model.py` does this with a blank tile). For many
  tiles write a builder, which calls the model by Ray handle inside the cluster.

## A new kind of model: an inference implementation

The rest of this guide is for a model that no existing implementation can serve:
export the model to ONNX, store it in MLflow, and then implement the Python
entrypoint that Ray Serve will load. Once the Python file is ready, continue with the [Deployment Guide](deployment-guide.md) for Helm configuration, deploy, and rollout monitoring.

## Export the Model to ONNX

Export your model to ONNX before writing the Serve entrypoint. The project expects ONNX so the runtime can load the artifact efficiently and use batching through `@serve.batch`.

If you already have a conversion script, reuse it here. Make sure the exported graph accepts a batch dimension, because the Ray Serve handler will batch multiple requests together.

```python
import torch

model = Model()
model.eval()

torch.onnx.export(
  model,
  torch.randn(1, 3, 512, 512),
  "model.onnx",
  export_params=True,
  do_constant_folding=True,
  input_names=["input"],
  output_names=["output"],
  dynamic_axes={
    "input": {0: "batch_size"},
    "output": {0: "batch_size"},
  },
)
```

If the exported model is too large and ONNX splits it into multiple files, see [Troubleshooting: Large ONNX Model Export](troubleshooting.md#large-onnx-model-export-2gb) for merge and multi-file handling.

## Upload the ONNX Artifact to MLflow

After export, upload the ONNX artifact to MLflow. If the export produced external weight files, upload the whole directory instead of only the `.onnx` graph file.

```python
import mlflow

with mlflow.start_run():
  mlflow.log_artifact("model.onnx", artifact_path="model")

  # If the export is split into multiple files, upload the full directory:
  # mlflow.log_artifacts("onnx_export_dir", artifact_path="model")
```

The deployment guide explains how the runtime reads this artifact back during startup.

## Start With an Existing Implementation

Do not start from an empty file. Copy the closest implementation and adapt only model-specific pieces.

- **Binary classification (baseline)**: [`src/model_service/model_implementations/binary_classifier.py`](https://github.com/RationAI/model-service/blob/main/src/model_service/model_implementations/binary_classifier.py)
- **Semantic segmentation**: [`src/model_service/model_implementations/semantic_segmentation.py`](https://github.com/RationAI/model-service/blob/main/src/model_service/model_implementations/semantic_segmentation.py)
- **Virchow2 embedding/classification**: [`src/model_service/model_implementations/virchow2.py`](https://github.com/RationAI/model-service/blob/main/src/model_service/model_implementations/virchow2.py)
Matching model files live in `deploy/helm/rayservice/models/`; the YAML itself is covered in the [Deployment Guide](deployment-guide.md).

## Python Entrypoint

Create a new file in `src/model_service/model_implementations/` (for example `src/model_service/model_implementations/my_model.py`). The sections below explain what each method does and what belongs inside it.

### `__init__`: one-time lightweight setup

```python
def __init__(self) -> None:
  import lz4.frame
  self.lz4 = lz4.frame
```

- `import lz4.frame`: imports compression utilities once for this replica.
- `self.lz4 = lz4.frame`: stores a reusable handle so request handlers do not re-import.

What belongs in `__init__`:

- lightweight constants,
- reusable helpers,
- cheap setup only.

What should not be here:

- heavy model download,
- config-dependent initialization.

### `reconfigure`: required runtime initialization

`reconfigure` is called after startup and on `user_config` changes. In this project, treat it as required because model path and runtime settings come from config.

```python
def reconfigure(self, config: Config) -> None:
  import importlib
  import onnxruntime as ort

  self.tile_size = config["tile_size"]

  model_config = dict(config["model"])
  module_path, attr_name = model_config.pop("_target_").split(":")
  provider = getattr(importlib.import_module(module_path), attr_name)

  self.session = ort.InferenceSession(
    provider(**model_config),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
  )

  self.input_name = self.session.get_inputs()[0].name
  self.output_name = self.session.get_outputs()[0].name

  self.predict.set_max_batch_size(config["max_batch_size"])
  self.predict.set_batch_wait_timeout_s(config["batch_wait_timeout_s"])
```


Data and control flow:

- Read static inference shape from config (`tile_size`).
- Resolve the artifact loader from `_target_` and fetch the ONNX artifact (typically from MLflow).
- Build ONNX Runtime session with desired execution providers.
- Cache input and output tensor names for fast inference calls.
- Apply batching limits from config directly to `predict`.

### `predict`: batched ONNX inference

```python
@serve.batch
async def predict(self, images: list[NDArray[np.uint8]]) -> list[float]:
  batch = np.stack(images, axis=0, dtype=np.uint8)
  outputs = self.session.run([self.output_name], {self.input_name: batch})
  return outputs[0].flatten().tolist()
```

What happens with data:

- Ray Serve collects many incoming requests into `images`.
- `np.stack` converts many single images into one batch tensor.
- ONNX runtime computes one forward pass over the whole batch.
- Output tensor is flattened and returned as a Python list.
- Ray Serve maps each list item back to the original HTTP request.

### `get_config`: expose model settings for builders

If your model serves the `spatial-scores` contract (used by the `heatmap` builder), you must provide a `get_config` method that builders can call through a Serve handle. The builder uses this to read `tile_size`, `output_tile_size`, `n_channels`, and `mpp` so it can pick the right tiling grid and resolution.

```python
async def get_config(self) -> dict[str, Any]:
  return {
    "tile_size": self.tile_size,
    "output_tile_size": self.output_tile_size,
    "n_channels": self.n_channels,
    "mpp": self.mpp,
  }
```

The contract's model client (`src/model_service/deployed_models/model_clients.py`) calls it with `await model.get_config.remote()` before each run and fails with `model_contract_mismatch` if the values differ from the model's catalog declaration; keep it cheap and avoid any I/O.

A `tile-embedding` model (used by the `tile-embeddings` builder) instead provides `get_config()` returning `tile_size` and `summaries`, and `async def embed(self, tile)`: one CHW uint8 tile in, float32 `(dim,)` (one summary) or `(len(summaries), dim)` out, with the model's own preprocessing and pooling inside. See `src/model_service/model_implementations/virchow2.py`. Builders call these methods by handle. The HTTP `root` route is optional: it is only for posting raw tiles to the model directly, and should call the same methods. `tests/deployed_models/test_catalog.py` fails if a model file's implementation lacks the methods of its contract.

### `root`: HTTP request parsing and serialization

```python
@fastapi.post("/")
async def root(self, request: Request) -> float:
  data = await asyncio.to_thread(self.lz4.decompress, await request.body())
  image = (
    np.frombuffer(data, dtype=np.uint8)
    .reshape(self.tile_size, self.tile_size, 3)
    .transpose(2, 0, 1)
  )
  result = await self.predict(image)
  return result
```

What each step does:

- `await request.body()`: gets raw HTTP bytes.
- `lz4.decompress`: reconstructs original image bytes.
- `np.frombuffer(...).reshape(...)`: converts bytes to `HWC` uint8 image.
- `.transpose(2, 0, 1)`: converts image to `CHW` layout expected by ONNX model.
- `await self.predict(image)`: sends item to batching queue and waits for matching output.

### Application binding

```python
app = MyModel.bind()
```

This exported symbol is what `import_path: models.my_model:app` points to in Helm.

### Using Foundation Models from Your Model

If you are deploying a model that is a downstream head (e.g., an MLP or Attention layer trained on top of a foundation model like Virchow2 or Prov-GigaPath), you **do not** need to re-export the entire foundation model into your ONNX artifact. 

Because foundation models are already deployed as independent services within the cluster, your model can directly invoke them via Ray Serve handles. In your `root` or `predict` method, call the foundation model first, then pass its output to your custom layers. If you call a foundation model's `predict` method directly, do **not** pass a raw `np.ndarray` image; first apply the same preprocessing/transforms that deployment expects, or call the model's ingress/request path instead.

```python
# In your __init__ or reconfigure method:
from ray import serve
self.foundation_model = serve.get_app_handle("virchow2")
self.foundation_transform = build_virchow2_transform()

# In your request handler:
@fastapi.post("/")
async def root(self, request: Request):
    # 1. Fetch raw image bytes from request and decode to an image / np.ndarray
    ...

    # 2. Apply the same transforms used by the foundation model deployment
    image_tensor = self.foundation_transform(image)
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # 3. Call the foundation model with the transformed tensor batch
    embedding = await self.foundation_model.predict.remote(image_tensor)
    
    # 4. Pass the embedding to your own model's predict endpoint
    return await self.predict(embedding)
```

## Whole-Slide (WSI) Inference and Builders

Builders in `src/model_service/builders/` do the whole-slide work; a model only answers per tile.

1. **Heatmaps and segmentation maps:** a `spatial-scores` model works with the
   `heatmap` builder, which tiles the slide at the model's resolution, averages
   overlapping predictions and writes a score grid (`.npz`). Nothing to implement.
2. **Tile embeddings:** a `tile-embedding` model works with the `tile-embeddings`
   builder, which writes one feature row per selected tile to HDF5.
3. **Other slide-level outputs** (a slide score, annotations, statistics) need a
   new builder, and a model that fits neither contract needs a new contract:
   see [Adding a builder or an inference contract](adding-builders.md).

## Next Step

List the model file under `models:` in the values file and `helm upgrade`; see
[Iterating on a model](#iterating-on-a-model) above. For how the service fits
together, read the [README](../../README.md); for EMPAIA, the
[EMPAIA integration](../architecture/empaia.md).
