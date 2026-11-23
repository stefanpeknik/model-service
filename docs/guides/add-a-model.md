# Add a model

A trained model that an existing runtime can run is one YAML file in
`catalog/served-models/`. You write no Python and rebuild no image. The
runtimes today are `onnx` for an ONNX graph and `timm` for a timm or
Hugging Face encoder; [Components](../reference/components.md) lists them with
every option. If your model needs a runtime, loader or step that doesn't exist
yet, write it first: [Write a component](write-a-component.md).

## 1. Copy the closest file

Start from the served model most like yours:

| Your model | Copy |
| --- | --- |
| ONNX classifier, one score per tile | `prostate-classifier-1.yaml` |
| ONNX segmentation, a score map per tile | `episeg-1.yaml` |
| Encoder on the Hugging Face Hub | `virchow2.yaml`, or `prov-gigapath.yaml` if it needs its own resize and normalisation |

```sh
cp catalog/served-models/prostate-classifier-1.yaml catalog/served-models/my-classifier.yaml
```

The file name is the model's ID and its route, so use lowercase letters,
digits and hyphens. Keep the first line. It points your editor at the file's
JSON Schema, so an editor with the YAML language server (VS Code's YAML
extension, for one) completes keys and underlines mistakes as you type.

## 2. Describe your model

```yaml
# yaml-language-server: $schema=../.schemas/served-models.schema.json
weights: {use: mlflow, uri: "mlflow-artifacts:/65/<run>/artifacts/model.onnx"}
runtime: {use: onnx, tensorrt: true, workspace_gib: 8, optimization_level: 3, threads: 4}
input: {tile_size: 512, mpp: 0.468}
output: {kind: scores, channels: 1, size: 1, range: [0, 1]}
batching: {max_size: 256, wait_s: 0.05}
resources: {gpus: 1, cpus: 4, memory_gib: 12}
```

- `weights` says where the weights come from. An MLflow URI names one run's
  file; add `sha256:` (the file's `sha256sum`) so overwritten weights are
  refused instead of served. A Hugging Face repository needs the full commit
  hash as `revision`, so the weights can't change under the same file.
- `input` is the tile edge in pixels and the microns per pixel the model was
  trained at. Get `mpp` right: analyses read the slide level closest to it.
- `output` is `{kind: scores, channels, size}` for a score map of
  `channels × size × size` per tile, or `{kind: embedding, summaries: [...]}`
  for an encoder. Declare `range` whenever you know it. A model that returns
  logits instead of probabilities then fails on its first tile, instead of
  drawing a heatmap that looks fine and means nothing.
- `pre` and `post` are steps run on each tile before and after the runtime,
  such as `resize`, `center_crop`, `normalize` or `cls_mean_pool`.
- `batching`, `resources` and the optional `serving` block size the Ray Serve
  deployment. [Served models](../reference/served-models.md) shows what every
  key means and what the four existing models use.

## 3. Check it

```sh
uv run pytest tests/catalog
```

This loads every catalog file for every environment, exactly as the API will,
and lists every mistake in one run. Each error names the file, the YAML path,
what's wrong, what would fit and how to fix it:

```text
catalog/served-models/my-classifier.yaml:runtime.thread: unknown option `thread`. Options of `onnx`: apply_session_options, fp16, optimization_level, output_dtype, tensorrt, threads, workspace_gib. Hint: did you mean `threads`?
```

Then regenerate the reference docs, which list your model, and commit them
with the file. `tests/test_generated.py` fails until you do.

```sh
PYTHONPATH=src uv run tools/generate.py
```

## 4. Test it

`uv run pytest tests/catalog` is the quick check while you edit. Before you
deploy, run the whole suite once:

```sh
uv run pytest
```

It needs the development environment from the README's
[Develop](../../README.md#develop) section: `uv sync` with the process
extras, and a PostgreSQL container if you want the scheduler tests to run
rather than skip. With `helm` installed, the suite renders the chart with
your model, and it checks the generated docs, so it catches mistakes the
catalog check can't see. Nothing else checks the file before the cluster
loads it.

## 5. Deploy it

A values file deploys every served model, or the IDs it names in
`servedModels`. `values.rationai-test.yaml` names them, so add yours there. If
the model differs on one cluster, say a smaller batch or another MLflow
server, put only the difference in
`catalog/environments/<environment>/served-models/my-classifier.yaml`. The
test cluster's `prostate-classifier-1.yaml` patch is one line. Then run
`helm upgrade`. A catalog change needs no image rebuild, and the API picks up
the new catalog without restarting. Running `helm upgrade` against a shared
cluster needs authorization for your task, as
[AGENTS.md](../../AGENTS.md) says.

## 6. Try it

`GET /v1/models/my-classifier` on the API returns the declaration as deployed
and its digest. To send one blank tile straight to the model's Ray Serve
route from an API pod:

```sh
oc exec -i deploy/model-service-api -- python - my-classifier 512 < tools/warm_model.py
```

The first request to a cold model waits for a GPU worker, which can take a
few minutes. The host checks every batch against the file, so a wrong tile
size, a NaN or a score outside `range` fails that request with the reason.
If the model sits in `UPSCALING`, the cluster has no free GPU for
`resources`.

To run your model over whole slides, give it an analysis:
[Add an analysis](add-an-analysis.md).
