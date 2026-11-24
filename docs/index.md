# Model Service

Model Service runs RationAI's pathology models on Ray Serve. A trained model
is a **served model**, a whole-slide job that calls models is an
**analysis**, and both are YAML files in the repo's `catalog/`, assembled
from reusable **components**. The control plane runs analyses synchronously or as
durable executions, and offers them to EMPAIA as apps.

## Start here

- To deploy a trained model: [Add a model](guides/add-a-model.md).
- To run models over whole slides: [Add an analysis](guides/add-an-analysis.md).
- When a piece is missing: [Write a component](guides/write-a-component.md).
- To call a model directly on one tile: [Raw inference](available-models.md).

The [reference](reference/components.md) lists every component, served
model, analysis and data format, generated from the code. The
[documentation map](README.md) lists the rest, and
[How model-service is put together](architecture/core.md) explains the
design.
