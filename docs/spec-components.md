# Spec: model-service rebuilt from components, served models and analyses

Status: implemented on the `unit-platform` branch, November 2026. It
supersedes the [core specification](spec.md) wherever they differ.
[How model-service is put together](architecture/core.md) describes the code
as built, and the [guides](guides/add-a-model.md) show how to use it.

Each decision below is traced by name to a ticket of the wayfinder map this
spec was assembled from. The map, its tickets and the prototypes it cites live
in the author's working notes, outside this repository; the spec holds the
plan and the tickets the reasoning.

## Problem Statement

Adding anything to model-service means writing and wiring code in several places.

A new model of a known kind is already YAML only. Anything else is not:
- **A model that needs Python** (a Hugging Face encoder, a graph that doesn't fit an existing class, a head on an encoder) means copying a 130–190-line Ray Serve class into the team's model code, then rebuilding the image. The copies drift: one of them silently ignores its ONNX Runtime options.
- **A new analysis** (today a "builder") is about 10 files and 5 wiring points, with roughly 350 lines of Python plus tests. The two existing builders duplicate a ~200-line tile loop.
- **A new EMPAIA output type** means five places that must agree: the builder-keyed EAD table, EAD generation, the publisher registry key, the gateway interface with its test fake, and the SDK.

Three students independently wrote the same 170-line class to put a small head on Virchow2 (upstream PRs #12–#14), each asking for a whole GPU slice for a CPU matmul.

Mistakes surface late or never:
- A bad model file is only a log line at API start, and the model silently disappears.
- Logits instead of probabilities, a wrong normalisation or a CPU fallback come back as a plausible-looking heatmap.
- The guides describe file formats that no longer exist.

Students and researchers who come to deploy a model or try an idea shouldn't need to read or change the core code, or learn its tribal knowledge.

## Solution

model-service is rebuilt around three concepts ("Core concept model"):
- **Components:** small reusable pieces of code, each written once. Loaders fetch weights; runtimes run them; steps transform data; sources produce items from a slide; sinks collect results into a **data format**; converters and publishers carry data in and out of EMPAIA.
- **Served models:** one generic Ray Serve host per trained model, assembled in YAML from a loader, a runtime and light pre/post steps. It declares what it accepts and returns, and keeps its own raw-inference route.
- **Analyses:** concrete YAML files describing one slide-in, results-out job: a source, steps (which may call several served models), sinks, and how the analysis is **exposed** (an EMPAIA app, standalone files).

A newcomer writes a served-model file and an analysis file in a top-level `catalog/`, using components by name. Code is only needed when a component is missing, and then it is one isolated component with its own test. Every catalog file is checked before deployment, with errors that name the file, the YAML path and how to fix it. Deploying a new catalog file needs no image rebuild.

The four run surfaces stay:
1. raw inference straight to a served model;
2. a synchronous analysis run in one HTTP call;
3. a scheduled, durable analysis run;
4. EMPAIA jobs.

**Version 1 is functional parity with today's model-service on the new design:**
- the four served models with unchanged raw routes;
- every builder × model run possible today, as four analyses (`prostate-heatmap`, `episeg-heatmap`, `virchow2-embeddings`, `prov-gigapath-embeddings`);
- the prostate heatmap's EMPAIA app.

The builders retire only once all four analyses pass verification.

## User Stories

### Students and researchers adding models

1. As a student with a new checkpoint of a known model, I want to add one served-model file, so that it is deployed without writing Python.
2. As a student, I want to copy an existing served-model file and change only the weights, so that I can start from something that already works.
3. As a student with an ONNX model, I want the file to say which runtime runs it and whether TensorRT is used, so that I don't have to configure ONNX Runtime by hand.
4. As a student with a Hugging Face encoder, I want to name the repository and a pinned revision in YAML, so that weights are reproducible and nobody copies loading code.
5. As a student, I want to declare the tile size, resolution and output my model has, so that the system checks every request against it.
6. As a student, I want to declare the output's value range, so that a model emitting logits instead of probabilities fails on its first batch instead of producing a misleading heatmap.
7. As a student, I want light pre- and post-processing (resize, crop, normalise, pooling) to be listed steps, so that I reuse tested code instead of writing transforms.
8. As a student, I want resources (GPUs, CPUs, memory) and batching as plain fields with defaults, so that I don't need to learn Ray Serve's options.
9. As a student, I want the TensorRT engine cache placed automatically, so that I never set a cache path or collide with another model's engine.
10. As a student, I want my served model to keep a raw-inference route, so that I can call it directly while experimenting.
11. As a student, I want a served model that was deployed with a configuration other than its catalog file to be refused, so that a stale deployment can't serve wrong results.
12. As a student, I want the served model to report which execution provider it runs on, so that a silent fallback from TensorRT to CPU is visible.

### Students and researchers adding analyses

13. As a researcher, I want to describe an analysis as a source, steps and sinks in one file, so that I can read the whole job at a glance.
14. As a researcher, I want to name a served model as a step, so that the analysis uses its outputs without knowing how it is deployed.
15. As a researcher, I want an analysis to call more than one served model, so that combinations like an encoder followed by a head are possible.
16. As a researcher, I want small learned heads to run as CPU steps inside the analysis, so that they don't hold a GPU slice.
17. As a researcher, I want the tile geometry taken from the first model step, so that tile size and resolution never drift from what the model expects.
18. As a researcher, I want an analysis without any model (e.g. a stain or QC measure), so that model-free ideas fit the same system.
19. As a researcher, I want the inputs an analysis needs to follow from its components, so that I never declare them twice.
20. As a researcher, I want to mark the region of interest as required, optional or ignored through the source's options, so that analyses like EPISeg can insist on it.
21. As a researcher, I want several sinks in one analysis, written all or nothing, so that a failed run never leaves half its outputs.
22. As a researcher, I want per-run settings (stride, background skipping, tissue fraction) to come from the components' declared options, so that callers can tune them and EMPAIA can offer them, without separate parameter declarations.
23. As a researcher, I want every runnable analysis to be a concrete file, so that it can be read, checked, versioned and exposed.

### Writing a missing component

24. As a student whose model needs a piece that doesn't exist, I want to write one small component in isolation, so that I don't touch the core.
25. As a student writing a component, I want a template for its kind, so that I know its interface.
26. As a student writing a component, I want fake slides, fake models and a helper to run my component, so that I can test it without a cluster.
27. As a student, I want conformance tests to run automatically for my component, so that it behaves like every other component of its kind.
28. As a student, I want my component to declare what it reads and adds, so that the checker can tell others when it doesn't fit their analysis.
29. As a student, I want my component to provide a fix hint, so that people who misuse it learn how to fix their file.
30. As any contributor, I want a component used in a slot it doesn't belong to (e.g. a step used as a runtime) to be refused with a hint, so that the folder layout's meaning is enforced.
31. As any contributor, I want components to live in folders named by where they can be used, so that I know which ones fit where.
32. As a future student, I want components written by earlier students to be listed and documented automatically, so that I can reuse their work.

### Checking and deploying

33. As a student, I want one test command that checks every catalog file for every environment, so that I find all my mistakes before deploying.
34. As a student, I want errors to name the file, the YAML path, what is wrong, what would fit and how to fix it, so that I can fix them without reading code.
35. As a student, I want my editor to autocomplete and flag errors in catalog files, so that I see mistakes while typing.
36. As a student, I want reference documentation of every component, served model, analysis and data format generated from the code, so that it never goes stale.
37. As a student, I want a new or changed catalog file to deploy with a Helm upgrade and no image rebuild, so that I can try it on the cluster quickly.
38. As a student, I want a live smoke test that calls my deployed served model and runs my analysis on a small sample slide, so that I can confirm it works on the cluster.
39. As an operator, I want each environment to deploy every catalog file by default or a listed subset, so that test and production differ only where they must.
40. As an operator, I want environment-wide settings (MLflow tracking URI, caches, storage roots) defined once, so that catalog files don't repeat them.
41. As an operator, I want a file that differs on one cluster to be a small patch of only the difference, so that copies don't drift.
42. As an operator, I want catalog changes to reach the API without restarting it, so that deploying a model doesn't kill running jobs.
43. As an operator, I want an invalid catalog change to be rejected while the previous catalog keeps serving, with the reason in health and logs, so that a typo can't take the service down.
44. As an operator, I want an analysis whose served model isn't deployed to be reported with the missing model named, so that I know why it's unavailable.
45. As an operator, I want deploying only served models to give plain raw inference without the analysis runner, so that a minimal deployment needs no special switch.

### Callers

46. As a caller, I want to list analyses with their inputs, outputs, formats, settings and the models they use, so that I know what I can run.
47. As a caller, I want to list served models with what they accept and return and which analyses use them, so that I can choose models.
48. As a caller, I want to run an analysis synchronously in one HTTP call, so that short jobs are simple.
49. As a caller, I want to schedule an analysis and poll, cancel or look up its execution, so that long jobs are durable.
50. As a caller, I want the request to name inputs, settings and output paths as today, minus the model, so that migrating is mechanical.
51. As a caller, I want each run to execute exactly the analysis that was validated when it was admitted, even if the catalog changes mid-run, so that retries are consistent.
52. As a caller, I want each execution to record the resolved analysis and its digest, so that I know exactly what produced a result.
53. As a caller, I want readers for every data format available from the client SDK, so that I can open outputs in a notebook without guessing their layout.
54. As a caller, I want outputs to carry a versioned media type, so that a reader refuses a layout it doesn't know.
55. As a caller using the `rationai` SDK, I want `pipelines`-style methods for analyses replacing the builder methods, so that my scripts keep working after a small update.
56. As a caller of raw routes, I want today's request and reply formats unchanged, so that existing clients and upstream tests keep working.

### EMPAIA and pathologists

57. As a pathologist, I want the prostate cancer app to keep working with the same modes, preprocessing triggers and viewer settings, so that the rebuild is invisible to me.
58. As an analysis author, I want to expose an analysis to EMPAIA by adding an `expose.empaia` section, so that one analysis becomes one app.
59. As an analysis author, I want to name the EMPAIA type for each output, so that the right publisher is picked automatically.
60. As an analysis author, I want an output EMPAIA can't carry (e.g. a 7-channel map or embeddings) to be refused at load with an explanation, so that I don't find out at finish.
61. As an analysis author, I want listing preprocessing on an analysis whose region of interest is required to be refused at load, so that I don't offer a mode that can't work.
62. As an analysis author, I want the EMPAIA app name validated against EMPAIA's limits at load, so that registration doesn't fail later.
63. As an analysis author, I want each analysis's triggers to belong to its own app only, so that a model used by two analyses doesn't start two preprocessing jobs per slide.
64. As an operator, I want apps for analyses that no longer exist to be retired at startup, so that stale apps don't linger in the marketplace.
65. As an EMPAIA user, I want inputs (slide, rectangle, parameters) converted according to their declared type, so that nothing is guessed from the data.
66. As a maintainer, I want new EMPAIA capabilities to arrive as new converters or publishers once the SDK supports them, so that the core doesn't change.

### Core developers and maintainers

67. As a maintainer, I want one generic served-model host instead of one class per model, so that a fix in model setup applies to every model.
68. As a maintainer, I want one generic analysis runner instead of a Serve app per builder, so that adding an analysis needs no Helm or code change.
69. As a maintainer, I want the scheduler, durable executions and provider finish semantics left as they are, so that the correctness work already done is kept.
70. As a maintainer, I want capacity limits per served model, with an analysis waiting until every model it uses has a free slot, so that GPU use stays bounded.
71. As a maintainer, I want model-free analyses to have their own capacity key, so that they neither queue behind a single shared lock nor are never claimed.
72. As a maintainer, I want the per-model capacity limit in the served-model file, so that it lives next to the model.
73. As a maintainer, I want existing executions migrated to the new identity, so that history stays readable.
74. As a maintainer, I want the API process to validate every catalog file without loading Ray, ONNX, torch or OpenSlide, so that the process boundaries hold.
75. As a maintainer, I want golden references recorded from today's deployment before anything changes, so that every migration step can prove its outputs are unchanged.
76. As a maintainer, I want each old model class compared bit for bit with the new host on stand-in weights in CI, so that the migration is verified without GPUs.
77. As a maintainer, I want today's processors compared bit for bit with the analysis runner on a fake slide, so that the rebuild of the tile loop is verified.
78. As a maintainer, I want EADs and EMPAIA publisher calls compared between old and new code, so that the EMPAIA app doesn't change by accident.
79. As a maintainer, I want an old class deleted only after both the local and the live comparison pass, so that nothing is removed before its replacement is proven.
80. As a maintainer, I want known bugs reproduced during the migration and fixed afterwards as separate changes, so that the comparison stays clean.
81. As a maintainer, I want a test to fail when the generated schemas or reference docs are stale, so that they never drift.
82. As an AI coding agent working in this repo, I want catalog errors in a stable, readable format and generated reference docs, so that I can add models correctly without tribal knowledge.

## Implementation Decisions

### Components ("Component mechanism")

- A component is a small class with a `name`, a nested Pydantic `Options` model and one kind-specific method set. There are seven kinds:

  | Kind | Method set |
  |---|---|
  | loader | fetch weights → local path |
  | runtime | load → batch function |
  | step | batch → batch |
  | source | open inputs → frame, keys, per-key load |
  | sink | open / add / close → artifact in a data format |
  | converter | EMPAIA input → typed input |
  | publisher | stored output → EMPAIA output via prepare/commit |

- Components self-register when their module is imported. A folder scan imports every component, and duplicate names fail at import. YAML refers to components only by name, never by import string. Components live only in this repo; outside code enters through a small component written here.
- Folders are split by where a component may be used:
  - served-model-only (loaders, runtimes);
  - shared steps (usable as a served model's pre/post and as an analysis step);
  - analysis-only (sources, sinks);
  - EMPAIA (converters, publishers);
  - standalone.

  Each kind knows its allowed slots, and use in another slot is a load error with a hint. Amendment from "Analysis": steps that declare `weights:` may use loaders too.
- A component's module imports only light dependencies at top level and imports heavy ones inside methods, so the API validates without Ray, ONNX, torch or OpenSlide. The process-boundary test extends to import every component.
- `Options` forbid unknown keys. Options marked `PerRun` may be overridden per run, and they are the API settings and EMPAIA parameters, under flat names. A name collision within one analysis is a load error.
- Components declare the columns they read and add: dtype and shape, with symbolic dimensions (an int, any, an option name, or a model field). An override hook covers outputs that depend on options or the model. Sinks also declare which fields of the source's one-row frame they read. A `hint()` hook turns check failures into how-to-fix text.
- The catalog, the reference docs and the JSON Schemas are generated from the component classes.

### Served models ("Served model")

- One Ray Serve deployment class hosts every served model. Its core is a plain class with a thin Serve wrapper. It owns batching, one handle method `infer(tile)` replacing today's `predict`/`embed`, the raw-inference route, per-batch input and output checks (shape, dtype, finite, declared range), and `describe()`.
- `describe()` replaces the `get_config` echo. It returns a digest of the resolved declaration plus the weights revision, and the active execution providers. The analysis runner compares the digest with the catalog, and a mismatch is `model_contract_mismatch`.
- `input` and `output` declarations replace the inference contract. `output.kind` (`scores` or `embedding`) types the model step's column in analyses.
- Day-one components:
  - loaders: `mlflow`, `huggingface` (revision required), `local`;
  - runtimes:
    - `onnx`: TensorRT on/off, workspace, optimisation level, fp16, threads; the cache directory is derived from model ID and digest; input and output names are read from the session;
    - `timm`: enumerated constructor options, autocast, pretrained transform;
  - steps: `cls_mean_pool`, `resize`, `center_crop`, `normalize`.

  New steps are added when a real model needs one.
- Placement rule: steps that reduce data (e.g. Virchow2's class token + patch-mean pooling) run in the served model's `post`, so only small outputs cross the handle.
- The four current models become served-model files that reproduce today's behaviour bug for bug, including Prov-GigaPath's centre crop and EPISeg's ignored ONNX Runtime options, which are fixed later. This shape was validated by the "Port real cases" prototype:

  ```yaml
  # catalog/served-models/virchow2.yaml (from the "Port real cases" prototype)
  weights: {use: huggingface, repo: paige-ai/Virchow2, revision: "<pin>"}
  runtime: {use: timm, num_classes: 0, mlp_layer: swiglu_packed, act_layer: silu, autocast: fp16, transform: pretrained}
  post: [{use: cls_mean_pool, registers: 4}]
  input: {tile_size: 224, mpp: 0.5}
  output: {kind: embedding, summaries: [global, local_mean]}
  batching: {max_size: 512, wait_s: 0.1}
  resources: {gpus: 1, cpus: 4, memory_gib: 8}
  ```

- Raw-route formats stay as today, derived from the declared output with no per-model code:
  - single-score output: a JSON float;
  - score map: LZ4 bytes;
  - embedding: LZ4 bytes with a shape header and a header that skips `post`.

  Unifying the format is a separate later change, together with an SDK update.

### Analyses ("Analysis")

- One file per analysis. The file name is its public ID, used for the route, executions and the EMPAIA app. Sections: `summary`, `source`, `steps`, `sinks`, `expose`.

  ```yaml
  # catalog/analyses/tissue-7class.yaml (upstream PR #13, from the "Port real cases" prototype)
  source: {use: slide_tiles, stride_fraction: 1.0, skip_background: true}
  steps:
    - {model: virchow2}
    - use: onnx_head
      weights: {use: mlflow, uri: "mlflow-artifacts:/104/…/linear_head.onnx"}
      channels: 7
  sinks:
    - {use: score_grid, output: tissue}
  ```

- Model steps resolve against the served-model catalog. Several are allowed. Something that runs on features is an analysis step, not a served model. The source takes its tile geometry from the first model step unless set explicitly, and conflicting geometry is a load error.
- All checks run at load: the column fold over source → steps → sinks, with model dimensions filled in from the catalog. Every analysis names its models, so nothing waits for a request.
- Steps map items 1:1, and sinks collect. Tile skipping stays inside the source, through the team's tile reader, and dropped tiles still count toward progress. Expand and reduce steps are added only when a component needs one.
- Each sink declares whether it needs ordered rows; the run is ordered if any sink does. Several sinks write all or nothing: temporary files, committed together only on success.
- There is no separate escape hatch. Anything that isn't a tile chain is a custom source or sink.
- One generic analysis runner executes any resolved analysis. Its core is a plain async function with a thin Serve wrapper, and it owns concurrency, cancellation, progress, per-step timing and failure codes.
- Day-one components: source `slide_tiles` (its region-of-interest option is required, optional or ignored, per "Port real cases"); sinks `score_grid` and `embedding_table`.

### Data formats ("Data formats")

- A data format is a named, versioned file shape declared in one light module: a vendor media type with a version parameter, a dataclass, write/read/check functions, and a docstring that becomes its reference page. A layout change bumps the version, and readers refuse versions they don't know.
- Sinks name the format they write, never an EMPAIA type. Publishers declare which format they read and which EMPAIA type they emit.
- Day one: `score_grid` (today's npz layout), `embedding_table` (today's HDF5 layout, documented for the first time) and `value` (for the existing primitive publisher). Other formats come with the SDK capabilities that need them.

### EMPAIA exposure ("EMPAIA adapters and exposure")

- One component per EAD type:
  - **publishers** (format in, EAD type out): they contribute their EAD fragment and prepare everything before committing;
  - **converters** (EAD type in, typed input out): they contribute their EAD fragment and materialise the input.

  The input's declared type selects the converter, so the gateway no longer guesses.
- EAD generation merges component fragments (including top-level classes and rendering when needed) with the namespace, modes, permissions and viewer configuration key. The builder-keyed EAD table and the not-offered list go away.
- The gateway shrinks to SDK call families: post an output body, upload a pixelmap pyramid, input reads, finalize and failure calls.
- The publication loop stays Python in the provider: validate, prepare all, commit in order, `outcome_unknown` after the first write, finalize, and workspace removal ordered as today.
- `expose.empaia` on the analysis holds name, short name, description, modes (standalone parameters are the `PerRun` options; preprocessing triggers), the input type mapping (defaults: slide → wsi, mask → rectangle), output types and viewer settings. Outputs not listed aren't published.
- Load-time checks:
  - publishers check bound shapes (e.g. a continuous pixelmap needs a one-channel score grid);
  - preprocessing is refused when a required input exists;
  - the app name is checked against EMPAIA's length limit.
- The namespace is derived from the analysis ID. Registration reconciles at startup: it registers or updates the current analyses and retires owned apps that no longer exist (or unlists them, if the marketplace mock can't delete).
- Day-one components: converters `wsi`, `rectangle`, `primitive`; publishers `primitive` and `continuous_pixelmap`. New EMPAIA capabilities land in the SDK first and arrive later as new components.

### Run surfaces and identity ("Run surfaces and identity")

| Surface | Route |
|---|---|
| raw inference | unchanged, on Ray Serve per served model |
| synchronous run | `POST /v1/analyses/{id}/run` |
| scheduled run | `POST /v1/analyses/{id}/schedule` (execution routes unchanged) |
| EMPAIA | `POST /v1/empaia/analyses/{id}` |
| catalog | `GET /v1/analyses`, `GET /v1/models` |

- The request is today's request minus the model: inputs, settings, output paths.
- At admission the API resolves the analysis (components, options with overrides, served-model declarations, digest) and stores the document on the execution. Every attempt sends that stored document to the runner, and it doubles as the provenance record.
- Schema migration: an execution's identity becomes the analysis ID, the analysis digest and the list of served models used, replacing the builder and model columns. Old rows map to the matching day-one analysis IDs, and unmapped pairs keep a legacy ID. The submission key is unchanged.
- Capacity: an execution is claimable only if every served model it uses has a free slot. Locks are taken per model in sorted order. The limit comes from the served-model file, with today's environment variable as the default. Model-free analyses use their analysis ID as the capacity key.
- The builders toggle goes. The catalog is whatever is deployed, and an analysis naming an undeployed model is a catalog error.
- The `rationai` client SDK gains analysis methods replacing the builder methods, and it ships the data-format readers.

### Deployment ("Deployment wiring")

- Catalog files live in a top-level `catalog/` at the repo root, with served models, analyses and per-environment patches. The Helm chart reads it through a symlink; tests and tools read it directly. If chart packaging rejects the symlink, a copy step fixes it.
- Helm discovers files by globbing, so no lists are kept. Each environment deploys everything or a named subset, and environment-wide settings live once in its values file. Patches are deep-merged over base files, and the copied test-instance model files go.
- `serveConfigV2` gets one generic Serve application per served model, plus one analysis-runner application sized by a runner block in the values file. The runner isn't rendered when no analyses are deployed.
- The catalog ConfigMap carries the resolved catalog. The checksum annotation that rolls the API goes. The API reloads the mounted catalog in place, validating it first, and keeps the old catalog on failure.
- An image rebuild is needed only for code (components). Catalog changes are Helm upgrades.

### Newcomer tooling ("Newcomer tooling")

- There is no CLI.
- A pytest catalog test runs every load-time check for every environment through the API's loader, and reports all errors in the standard format: file, YAML path, what's wrong, what would fit, a hint.
- A fast live loop replaces local runs: catalog edits deploy without rebuilds, the host's guards fail loudly on the first request, and a live smoke test exercises a deployed served model or analysis.
- JSON Schemas for catalog files and the reference docs are generated by one script, and a test fails when they're stale. The hand-written guides shrink to three: add a model, add an analysis, write a component.
- Testing helpers: fake slide, fake model, run-a-component, and per-kind conformance suites.
- The local install extra leaves out the EMPAIA SDK, and CI must be able to install the SDK without SSH.

### Version 1 scope (map Notes, "Assemble spec and build backlog")

- **Parity:**
  - the four served models (`prostate-classifier-1`, `episeg-1`, `virchow2`, `prov-gigapath`) with unchanged raw routes;
  - the analyses `prostate-heatmap`, `episeg-heatmap` (optional region of interest, as today), `virchow2-embeddings` and `prov-gigapath-embeddings`;
  - the prostate heatmap's EMPAIA app;
  - the new run surfaces, migration and SDK update;
  - `catalog/` with Helm wiring;
  - the catalog test, schemas and generated docs;
  - all verification gates.
- The builders and the old model classes retire only after their replacements pass verification.

## Testing Decisions

- **A good test** exercises behaviour through a seam: the outputs a caller or deployment would see. It never tests internal structure. Equivalence tests compare outputs, not code paths.
- **Seams (agreed with the owner):**
  1. **Catalog loader:** a catalog directory plus an environment gives a resolved catalog or a list of errors. It covers all load-time checks, error format and hints. The pytest catalog test lives here.
  2. **Served-model host core:** built from a declaration, exercised through `infer`, `describe` and the raw-route reply. It covers component wiring, guards, digest and reply formats.
  3. **Analysis runner core:** a resolved document plus inputs plus model handles gives outputs. It covers sources, steps, sinks, ordering, all-or-nothing writes, progress, cancellation and failure codes.
  4. **HTTP API**, through the FastAPI test client and the PostgreSQL suite: routes, admission storing the resolved document, identity migration, per-model capacity.
  5. **EMPAIA provider with the fake gateway:** converters, publishers, EAD generation, exposure, registration reconcile.
  6. **Live cluster** (`pytest -m live`): golden references.

  Component conformance suites run through the component registry, under seams 1 and 3.
- **Verification ("Verification strategy"):**
  - **Step 0:** before any code changes, record golden references on the current deployment: about 32 tiles per served model through today's raw routes, plus synchronous runs of each day-one analysis's builder × model pair on 2 small slides. They're stored on shared storage, with a hash manifest in the repo.
  - **In CI (CPU, stand-in weights: tiny ONNX graphs, small random ViTs), outputs must be bit-identical:**
    - each old model class against the host, raw-route replies included;
    - today's processors against the runner, with one tile in flight (a concurrent run passes within `atol=1e-7`, the existing nondeterminism);
    - EADs (apart from the documented ID change) and fake-gateway publisher calls.
  - **Live, after each migration deploys:** ONNX scores within `atol=1e-3` with an identical scored mask, embeddings with cosine ≥ 0.9999 per tile, geometry exact.
  - An old class, its test shim and the equivalence test that imports it are deleted only after both checks pass. The golden references stay as the regression suite.
- **Prior art:**
  - API route tests in `tests/api/`;
  - the PostgreSQL scheduler suite (needs `TEST_POSTGRES_DSN`);
  - EMPAIA provider tests with the fake gateway in `tests/providers/` and `tests/support/empaia_fixtures.py`;
  - builder contract tests in `tests/support/builder_contract.py`;
  - the model-file test walking every shipped file;
  - the process-boundary test;
  - the chain-core and local-probe prototypes (bit-identical runner, seeded model mistakes);
  - upstream PR #10's snapshot tests and reference generator.

## Out of Scope

- New EMPAIA capabilities (annotations, classes, collections, multi-channel or nominal pixelmaps, polygon inputs) and their data formats. They land in the SDK first and come later as new components.
- Scheduler and API internals beyond what the run surfaces and identity require.
- Runtime reshaping unrelated to the generic host and runner: runs as Kubernetes Jobs, multiplexed model pools, Ray Data, runner pools per analysis size.
- Security hardening (NetworkPolicy, Ray auth, API authentication): a separate effort.
- A CLI and a local `try` runner. The runner and host keep plain cores so a script can be added later.
- Unifying raw-route wire formats.
- Outside-repo component packages (entry points).
- Thesis writing and tag timing.
- **After version 1, on the same design:** EPISeg with a required region of interest exposed in EMPAIA, heads via an `onnx_head` step (upstream PRs #12–#14), and the deferred bug fixes with their re-baselines (EPISeg's ONNX Runtime options; Prov-GigaPath served under a new ID with 256 px tiles).

## Further Notes

- **"CI" in this spec means the local test suite** (`uv run pytest`, PostgreSQL via `TEST_POSTGRES_DSN`). GitHub CI is enabled only once the work is approved and finalized (owner, 2026-11-22).
- **Vocabulary changes:** "pipeline" became **analysis**, matching EMPAIA's buttons for starting standalone jobs, and "unit" became **component**. Glossary changes on landing: add Component (with its kinds), Served model, Analysis, Sink, Data format, Exposure; retire Builder and Inference contract. "Workflow" stays retired. "App" stays EMPAIA's and Ray Serve's word.
- **Docs:**
  - generated reference docs replace hand-written component and format descriptions;
  - three guides replace `adding-models` and `adding-builders`;
  - the README and architecture docs are rewritten for the new concepts;
  - dated operations records and `docs/history/` stay as records;
  - the current `docs/spec.md` is marked superseded by this spec;
  - live validation of the rebuild gets its own dated operations file.
- **Evidence and prototypes:**
  - the map's research reports are on the local branches `research/contact-surface-inventory` and `research/provider-io-inventory`;
  - the "Port real cases" catalog files, in the map's `prototypes/port-real-cases/`;
  - the earlier exploration, `config-composition/` beside the map (`SYNTHESIS.md`, the chain-core and local-probe prototypes).
- **Known risks to watch during the build:**
  - whether the EMPAIA marketplace mock supports deleting apps;
  - Helm packaging with a symlinked `catalog/`;
  - CI access to the EMPAIA SDK;
  - TensorRT fp16 variation in the live tolerances.
