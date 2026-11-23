"""Helpers for testing a component in isolation, without a cluster.

- `fake_slide(path)`: a small tiled TIFF that OpenSlide and pyvips open;
- `fake_frame()`: the frame row a slide source gives, for sinks;
- `fake_model(output=…)`: a served-model stand-in for `model.` dimensions,
  batch calls and the host's `infer(tile)`;
- `run_component(component, data)`: run a step, source or sink once;
- `FakeGateway(directory)`: a fake EMPAIA job for converters and publishers;
- `conformance.check(component, work_dir)`: the suite of the component's kind.

Heavy libraries (pyvips) are imported only when a helper needs them.
"""

from components.testing import conformance
from components.testing.empaia import FakeGateway
from components.testing.models import FakeModel, fake_model
from components.testing.running import SourceRun, fake_batch, run_component
from components.testing.slides import fake_frame, fake_slide


__all__ = [
    "FakeGateway",
    "FakeModel",
    "SourceRun",
    "conformance",
    "fake_batch",
    "fake_frame",
    "fake_model",
    "fake_slide",
    "run_component",
]
