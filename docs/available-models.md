# Raw inference

Every served model answers single tiles on its own Ray Serve route,
`POST /<model-id>/` on the Ray Serve origin (`http://<release>-serve-svc:8000`
inside the cluster). The control plane doesn't proxy these calls; use an
[analysis](reference/analyses.md) to run a model over a whole slide. Which
models a deployment serves is in `GET /v1/models` on the control plane, and
[Served models](reference/served-models.md) lists the ones in `catalog/`.

## Request

The body is one tile, uint8 RGB in `(tile_size, tile_size, 3)` order
(channels last), compressed with LZ4's frame format. `tile_size` is the
model's `input.tile_size`, which `GET /v1/models/<id>` returns under
`declaration.input`.

## Reply

The format follows from the served model's `output:`:

| `output:` | Reply |
| --- | --- |
| `scores`, one channel of size 1 (`prostate-classifier-1`) | The score as a JSON float. |
| `scores` with a larger map (`episeg-1`) | The scores' raw bytes in the declared `dtype`, LZ4-compressed, shaped `(channels, size, size)`. |
| `embedding` (`virchow2`, `prov-gigapath`) | The features' raw bytes, LZ4-compressed, with their shape in the `x-output-shape` header. |

An embedding request may send two headers:

- `x-output-dtype: float16` casts the features (default `float32`);
- `x-pool-tokens: false` skips the model's `post:` steps. Virchow2 then
  returns its class token and patch tokens instead of the two pooled
  vectors, without its register tokens.

A tile of the wrong size, or an output that breaks the declaration (wrong
shape, NaN, a score outside the declared `range`), fails the request with the
reason.

## What a replica runs

`GET /<model-id>/describe` on the same origin answers what a replica loaded:

```json
{
  "id": "prostate-classifier-1",
  "digest": "5f0c…",
  "weights_revision": "mlflow-artifacts:/65/…/artifacts/model.onnx@sha256:9a1e…",
  "execution_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
}
```

`execution_providers` lists the providers ONNX Runtime actually uses, so a
model declared with `tensorrt: true` that lost TensorRT shows it here (and
the replica logs a warning when it starts). Replicas also log this line when
they are built. The call wakes a model scaled to zero, like any request.

## Example

```python
import ast

import httpx
import lz4.frame
import numpy as np

tile = np.zeros((224, 224, 3), dtype=np.uint8)  # virchow2 takes 224 px tiles
response = httpx.post(
    "http://model-service-serve-svc:8000/virchow2/",
    content=lz4.frame.compress(tile.tobytes()),
    headers={"x-output-dtype": "float16"},
    timeout=600,
)
response.raise_for_status()
shape = ast.literal_eval(response.headers["x-output-shape"])
features = np.frombuffer(lz4.frame.decompress(response.content), np.float16).reshape(shape)
print(features.shape)  # (2, 1280): the class token and the patch mean
```

The first request to a model scaled to zero waits for a GPU worker, which
can take a few minutes. `tools/warm_model.py` sends one blank tile to warm
a model. Concurrent requests to a batching model can differ slightly from a
request run alone (Virchow2: up to about 0.1%), because Ray batches them
together.
