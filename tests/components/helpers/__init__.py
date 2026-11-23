"""Helpers for testing a component in isolation, without a cluster.

- `fake_slide(path)`: a small tiled TIFF that OpenSlide and pyvips open;
- `fake_frame()`: the frame row a slide source gives, for sinks;
- `fake_model(output=…)`: a served-model stand-in for `model.` dimensions,
  batch calls and the host's `infer(tile)`;
- `run_component(component, data)`: run a step, source or sink once;
- `FakeGateway(directory)`: a fake EMPAIA job for converters and publishers;
- `conformance.check(component, work_dir)`: the suite of the component's kind;
- `offline.LOADERS`, `offline.RUNTIMES`: what the suite needs, by component
  name, to run a loader or runtime without a network.

Heavy libraries (pyvips, onnx, timm) are imported only when a helper needs them.
"""

from tests.components.helpers import conformance, offline
from tests.components.helpers.empaia import FakeGateway
from tests.components.helpers.models import FakeModel, fake_model
from tests.components.helpers.running import SourceRun, fake_batch, run_component
from tests.components.helpers.slides import fake_frame, fake_slide


__all__ = [
    "FakeGateway",
    "FakeModel",
    "SourceRun",
    "conformance",
    "fake_batch",
    "fake_frame",
    "fake_model",
    "fake_slide",
    "offline",
    "run_component",
]
