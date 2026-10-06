# Workflow definitions

The prostate heatmap definition is the runtime input/output contract.

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
The generation tests check workflow names and references; they do not validate
the external EMPAIA schema or register an application. Real EMPAIA integration
remains a separate phase.
