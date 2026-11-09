# Adding a builder or an inference contract

A model answers one tile at a time. A builder turns a whole slide into an
output by calling a model tile by tile. The two meet in an **inference
contract**: what a model of that kind takes and returns. A builder accepts one
contract, and any model of that contract can run with it.

```text
deploy/helm/rayservice/models/<model>.yaml    inference.contract: spatial-scores
        │ catalog (facade)                    serve: the Ray application
        ▼
builders/model_clients.py                     SpatialScoresClient: get_config + predict,
        │ client per contract                 checked answers, failure codes
        ▼
builders/<builder>/declaration.py             accepts=SPATIAL_SCORES, inputs, configuration,
builders/<builder>/builder.py                 outputs; run(context) → outputs
        │ registry.py lists it
        ▼
deploy/helm/rayservice/values.yaml builders   one Ray application per builder
```

Adding a model of an existing contract needs only YAML; see
[Adding models](adding-models.md). A new kind of slide output needs a builder.
A model that takes or returns something no contract describes needs a
contract first.

## A builder for an existing contract

Take a builder that traces annotations from a `spatial-scores` model.

### 1. Declare it

`src/model_service/builders/annotations/declaration.py` holds everything the
facade needs: which contract the builder accepts, its inputs, its configuration
and its outputs. It imports only Pydantic and `builders.spec`, because the
facade loads it without the slide libraries.

```python
class AnnotationsInputs(BuilderInputs):
    slide: Slide
    region_of_interest: RegionOfInterest = None


class AnnotationsConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold: Annotated[float, Field(ge=0, le=1)] = 0.5


class AnnotationsOutputs(BuilderOutputs):
    annotations: Annotated[
        SharedArtifact,
        Artifact(media_type="application/geo+json", default_path="annotations.geojson"),
    ]


SPEC = BuilderSpec(
    id="annotations",
    accepts=InferenceContract.SPATIAL_SCORES,
    input_model=AnnotationsInputs,
    output_model=AnnotationsOutputs,
    configuration=AnnotationsConfiguration,
    implementation="model_service.builders.annotations.builder:AnnotationsBuilder",
)
```

- **Inputs:** a required input is a `Path` and an optional one is
  `Path | None = None`. `Slide` and `RegionOfInterest` are the shared
  annotated types, and `Input(media_type=...)` declares another input.
- **Outputs:** an output is a `SharedArtifact` (a file, annotated with
  `Artifact`) or an `InlineOutput` (a JSON value in the result).
- **Configuration:** its defaults are what callers get, and its field names
  are what EMPAIA jobs and standalone requests may override.

### 2. Implement it

`builder.py` subclasses `Builder[Client, Inputs, Configuration, Outputs]`. The
client type is the contract's client from `model_clients.py`.

```python
class AnnotationsBuilder(
    Builder[SpatialScoresClient, AnnotationsInputs, AnnotationsConfiguration, AnnotationsOutputs]
):
    def __init__(self, settings: WorkerSettings) -> None:
        self._settings = settings

    async def run(self, context) -> AnnotationsOutputs:
        with context.write_artifact("annotations") as annotations:
            await trace_annotations(
                context.inputs.slide,
                context.inputs.region_of_interest,
                annotations.path,
                client=context.model,
                threshold=context.configuration.threshold,
            )
        return AnnotationsOutputs(annotations=annotations.artifact)
```

By the time `run` is called, the application has already:

- checked the invocation's inputs and the model's contract;
- checked that the running model matches its catalog entry (`get_config`);
- resolved every input to a local file (`context.inputs`);
- validated the configuration (`context.configuration`).

The builder's own job is to:

- **Read the slide** through `builders/slides/`: `open_slide` and `TileReader`
  read tiles at the model's resolution, skipping background and tiles outside
  the region of interest. Grid placement comes from `ratiopath`, as in the
  existing builders.
- **Call the model** through `context.model`, the contract's client, whose
  `context.model.declaration` is the catalog entry. Never call Ray directly:
  the client classifies failures and checks answers the same way for every
  builder.
- **Write each artifact** inside `context.write_artifact(name)`. It writes to
  a temporary file and replaces the destination only on success, so a failed
  run leaves nothing behind. It also refuses an output path that would
  overwrite an input.
- **Return its outputs model.** The application checks it against the
  declaration and stamps the declared media types on it.

Raise `InvalidInvocation` for bad input and `BuilderFailure` for anything with
a failure code. Anything else becomes a retryable internal error.

### 3. Register and deploy it

- Add `SPEC` to `BUILDERS` in `src/model_service/builders/registry.py`.
- Add an entry under `builders:` in `deploy/helm/rayservice/values.yaml`, keyed
  by the builder ID, with its Ray deployment options and `userConfig`
  (`num_threads`, `max_concurrent_tasks`). The chart makes it a Ray application
  at `/builders/<id>` running `builders.serve:app`, and tells the facade it
  exists.

The builder then gets the public routes (`/v1/builders/<id>/models/<m>/run`
and `/schedule`), scheduling, retries and the SDK with no further code.

### 4. Offer it in EMPAIA (optional)

Add the builder's EAD `io` entries to `BUILDER_IO` in
`src/model_service/providers/empaia/io.py`. Each output's EMPAIA type needs a
publisher in `providers/empaia/publishers.py` for its (type, media type). An
app whose outputs nothing can publish is not registered, and the reason is
logged.

### 5. Test it

`tests/builders/<builder>/test_builder.py` subclasses `BuilderContractTests`:

```python
class AnnotationsContractTests(BuilderContractTests, unittest.IsolatedAsyncioTestCase):
    builder_id = "annotations"
    model = SpatialScoresModel(id="m", tile_size=32, output_tile_size=4, n_channels=1, mpp=0.5)
```

That class runs the builder on a real slide against a fake model and checks
that:

- the outputs are the declared ones, written at their default paths, with
  nothing left behind;
- `output_path` places the artifact;
- an unreadable slide is an invalid input;
- a model that differs from its declaration is never called;
- a failing model leaves no artifact.

Add tests for what the builder computes. `tests/builders/test_registry.py`
checks that the builder's type arguments match its declaration and that every
builder in `values.yaml` is registered.

## A new inference contract

A contract is needed when a model takes or returns something no existing client
describes, for example a whole region instead of a tile, or a mask instead of
scores.

1. **Declare its models.** In `src/model_service/builders/contracts.py`:
   - add a value to `InferenceContract`;
   - add a model declaration class with the fields builders need, such as
     `tile_size` and `mpp`;
   - add that class to `ModelBinding`.

   A model file's `inference` block must then match it (see
   [Adding models](adding-models.md)).
2. **Write its client** in `src/model_service/builders/model_clients.py`, a
   `ModelClient` subclass. It sets `contract`, and `methods`: the deployment
   methods a model of the contract must have. It also implements:
   - `_matches(running)`: does the `get_config()` result equal the
     declaration?
   - one method per model call, which uses `_call` (it turns failures into
     `model_failed`) and checks the answer, raising `_invalid_output()` when
     the answer breaks the contract.

   Add the client to `CLIENTS`.
3. **Implement it in models.** A model implementation in `src/model_service/models/`
   provides those methods, and its YAML names the contract.
   `tests/facade/test_model_catalog.py` fails if a model file's implementation
   lacks a method its contract's client calls.

Then add builders that accept the new contract, as above.
