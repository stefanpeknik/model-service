# Workflow definitions

The prostate heatmap definition is the runtime input/output contract.

- `../helm/rayservice/applications/prostate-classifier-1.yaml` owns the model
  artifact, tile size, training MPP, Ray resources, batching, and autoscaling.
- `prostate-heatmap.yaml` owns the stable workflow identifier and its named
  input/output contract. It does not select model deployments. The Python
  workflow may invoke zero, one, or several model deployments.
- The optional `empaia` metadata is reserved for later EAD generation from the
  same input/output contract.
- JES deployment configuration will own the model-service address. It is not
  an EAD field.

The model's `mpp` is fixed inside model-service for both RationAI SDK and
EMPAIA calls. It is not a per-EMPAIA-job request or EAD option. Tile size also
comes from the model deployment; neither value is copied into the workflow
definition. How a workflow should handle a WSI whose available levels differ
from the target MPP is TBD pending discussion with the model supervisor. No
resampling behavior is assumed.
