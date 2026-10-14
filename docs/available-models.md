# Available Models & Endpoints

This document describes implemented model endpoints and their request and response
formats. It is not a live deployment inventory; enabled applications depend on
the deployment profile. For builders and durable executions, see
[the architecture](architecture/core.md).

## Overview of Endpoints

All endpoints receive and return data over HTTP using `POST` requests. To minimize network overhead, image payloads and tensor outputs are compressed using **LZ4**.

| Application Name          | Route Prefix             | Model Type                    |
| ------------------------- | ------------------------ | ----------------------------- |
| **Prostate Classifier 1** | `/prostate-classifier-1` | Binary Classification         |
| **Episeg 1**              | `/episeg-1`              | Semantic Segmentation         |
| **Virchow2**              | `/virchow2`              | Foundation Model / Embeddings |
| **Prov-GigaPath**         | `/prov-gigapath`         | Foundation Model / Embeddings |

---

## Model Specifications

### 1. Binary Classifier (`/prostate-classifier-1`)

A binary classification model (e.g., tumor vs. normal tissue).

- **Input**: LZ4-compressed raw bytes of an image (RGB format).
  - The model converts these bytes back to a NumPy array (`uint8`) of shape `(tile_size, tile_size, 3)`.
- **Output**: A single floating-point number representing the classification score.

**SDK example**:

```python
from rationai import Client

with Client() as client:
    score = client.models.classify_image(model="prostate-classifier-1", image=image, timeout=30.0)
    print(f"Classification score: {score}")
```

### 2. Semantic Segmentation (`/episeg-1`)

A semantic segmentation model yielding a prediction map over the input image.

- **Input**: LZ4-compressed raw bytes of an image (`uint8`).
  - Expected layout before compression is `(tile_size, tile_size, 3)`.
- **Output**: LZ4-compressed sequence of bytes representing an `np.float16` NumPy array.
  - The client SDK should decompress this buffer and reconstruct the float16 array.
- **SDK example**:

```python
from rationai import Client

with Client() as client:
    seg = client.models.segment_image(model="episeg-1", image=image, timeout=30.0)
    # `seg` is returned as a NumPy array (float16 when configured). SDK handles LZ4 decompression.
    print(seg.shape)
```

### 3. Virchow2 (`/virchow2`)

A versatile foundation model (Virchow2) used primarily for generating embeddings or classification features.

- **Input**: LZ4-compressed raw bytes of a tissue tile image (`uint8`, shape `(tile_size, tile_size, 3)`).
- **Output**: Output tensor matching the user's requested specification.
- **Headers**:
  - `x-output-dtype` (optional, default: `float32`): Sets the return precision. Can be `float32` or `float16`.
  - `x-pool-tokens` (optional, default: `true`): If `true`, returns a pooled result (usually `class_token` and `mean` patch tokens). If `false`, returns unpooled raw outputs.
- **SDK example**:

```python
from rationai import Client
import numpy as np

with Client() as client:
  emb = client.models.embed_image(
    model="virchow2",
    image=image,
    output_dtype=np.float16,
    timeout=30.0,
    pool_tokens=False,
  )
  print(emb.shape)
```

### 4. Prov-GigaPath (`/prov-gigapath`)

A foundation model for pathology tile embeddings (Prov-GigaPath).

- **Input**: LZ4-compressed raw bytes of a tissue tile image (`uint8`, shape `(tile_size, tile_size, 3)`).
- **Output**: Output tensor matching the user's requested precision.
- **Headers**:
  - `x-output-dtype` (optional, default: `float32`): Sets the return precision (for example `float16`).
- **SDK example**:

```python
from rationai import Client
import numpy as np

with Client() as client:
  emb = client.models.embed_image(
    model="prov-gigapath",
    image=image,
    output_dtype=np.float16,
    timeout=30.0,
  )
  print(emb.shape)
```

### 5. Builders (`/v1/builders/...` on the facade)

Builders run a model over a whole slide. They are called through the facade's
public API, not on the Ray route, and take the model as a parameter:

| Builder | Accepts | Output |
| --- | --- | --- |
| `heatmap` | `spatial-scores` models (classifiers, segmentation) | pyramidal TIFF score map |
| `tile-embeddings` | `tile-embedding` models (Virchow2, Prov-GigaPath) | HDF5 with one feature row per tile |

`GET /v1/models` lists the deployed models and the builders each works with.

```python
from rationai import Client
from rationai.types import BuilderRequest, InputAccess, SharedPathInput

with Client(base_url="http://<facade>:8080") as client:
  result = client.builders.run(
    "heatmap",
    BuilderRequest(
      model_id="prostate-classifier-1",
      inputs={
        "slide": SharedPathInput("slides/slide.mrxs"),
        "region_of_interest": SharedPathInput("masks/mask.tif"),  # optional
      },
      input_access=InputAccess("shared_storage", {"root": "/mnt/data"}),
      output_path="outputs/heatmap.tif",
    ),
  )
```

---

## SDK Integration Patterns

The RationAI SDK provides convenient methods for interacting with each model type. All image data is automatically compressed with LZ4 before transmission, and responses are automatically decompressed.

### Using the RationAI Client

Initialize the client and call model methods:

```python
from rationai import Client
from PIL import Image
import numpy as np

with Client() as client:
  image = Image.open("tissue_sample.tiff")

  # Binary classification
  score = client.models.classify_image(model="prostate-classifier-1", image=image)
  print(f"Classification score: {score}")

  # Semantic segmentation
  segmentation = client.models.segment_image(model="episeg-1", image=image)
  print(f"Segmentation shape: {segmentation.shape}")  # (num_classes, height, width)

  # Embedding with custom options
  embedding = client.models.embed_image(
    model="virchow2",
    image=image,
    output_dtype=np.float16,
    pool_tokens=False,
  )
  print(f"Embedding shape: {embedding.shape}")
```
