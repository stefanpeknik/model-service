# Workflow definitions

The YAML definitions are runtime input/output contracts. Besides prostate heatmap,
they declare EPISeg segmentation and GigaPath/Virchow2 embedding workflows.
The [additional-workflow guide](../docs/guides/additional-workflows.md) describes
their output formats and deployment requirements. These new definitions accept
shared-storage inputs only; EMPAIA metadata and publication are deferred.

- `../helm/rayservice/applications/prostate-classifier-1.yaml` owns the model
  artifact, tile size, training MPP, Ray resources, batching, and autoscaling.
- `prostate-heatmap.yaml` owns the stable workflow identifier and its named
  input/output contract. It does not select model deployments. The Python
  workflow may invoke zero, one, or several model deployments.
- The optional `empaia` metadata supports EAD generation from the same
  input/output contract. Local generation tooling is not EAD registration or
  proof that JES routing is integrated.
- JES deployment configuration will own the model-service address. It is not
  an EAD field.

The model's `mpp` is fixed inside model-service for both RationAI SDK and
EMPAIA calls. It is not a per-EMPAIA-job request or EAD option. Tile size also
comes from the model deployment; neither value is copied into the workflow
definition. The current heatmap processor selects the closest available slide
level and uses its actual resolution metadata. Model-owner approval of the
level-selection and resampling policy is still pending; this implementation
choice does not settle that question.

## Local EAD generation

Generate the draft EAD from the workflow's optional EMPAIA metadata:

```sh
UV_CACHE_DIR=.cache/uv uv run --no-sync python scripts/generate_ead.py \
  workflow-definitions/prostate-heatmap.yaml eads/prostate-heatmap.json
```

Add `--check` to verify that the checked-in JSON matches without writing it.
Both generation and checking validate against the official Draft 7 schema in
[`../eads/schemas/ead-v3.schema.json`](../eads/schemas/ead-v3.schema.json),
without network access. Invalid definitions are rejected before writing output.
The generation tests also check workflow names and references. These checks do
not register an application or prove compatibility with the deployed AppService.
Real EMPAIA integration remains a separate phase.

The schema is an unchanged copy from EMPAIA's definitions repository at commit
[`2b324f13f73eaa5950794ce9befeffa4769e8a78`](https://gitlab.com/empaia/integration/definitions/-/raw/2b324f13f73eaa5950794ce9befeffa4769e8a78/ead/ead-schema.v3.json).
Its upstream MIT license is preserved in `../eads/schemas/LICENSE`. To update it,
choose the version supported by the target EMPAIA installation, replace the copy
from a pinned upstream commit, update this reference, and rerun the generation
tests and `--check`. Do not hand-edit the upstream schema.
