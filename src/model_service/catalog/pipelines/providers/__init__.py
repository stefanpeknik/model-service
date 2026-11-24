"""A pipeline's `providers.empaia`: the one EMPAIA app the pipeline becomes.

```yaml
providers:
  empaia:
    name: Prostate cancer heatmap     # the app's name, at most 30 characters
    name_short: ProstHM
    description: Classifies prostate tissue and shows the scores as a heatmap.
    modes:
      standalone: {parameters: [stride_fraction]}   # settings users fill in
      preprocessing: {triggers: [{tissue: PROSTATE, stain: H_AND_E}]}
    inputs:                           # optional: slide is wsi, region_of_interest rectangle
      region_of_interest: {type: rectangle}
    outputs:
      heatmap: {type: continuous_pixelmap}
    viewer: {outputs: {heatmap: {render: shader, …}}}
```

Each input's `type` picks its converter, and each output's `type` with its
sink's format picks its publisher; other keys are that component's options.
Sink outputs not listed aren't published. Loading checks that every input has
a converter and exactly one is the slide, that preprocessing (started with a
slide alone) needs no other input, that parameters are plain per-run
settings, that each output has a publisher that takes what its sink writes,
and EMPAIA's limits on names.

The package splits by concern: `fields` holds the keys a file may write,
`schema` narrows them to the EAD types the registry's components handle, and
`resolution` picks the converters and publishers and runs the checks.
"""

from model_service.catalog.pipelines.providers.resolution import (
    EmpaiaPipelineSettings,
    resolve_providers,
)
from model_service.catalog.pipelines.providers.schema import providers_schema


__all__ = ["EmpaiaPipelineSettings", "providers_schema", "resolve_providers"]
