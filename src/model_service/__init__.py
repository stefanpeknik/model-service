"""RationAI model service: runs pathology models on Ray Serve behind one API.

This package is the plumbing. The components it runs (steps, loaders,
runtimes, sources, sinks, EMPAIA converters and publishers) and the classes
they subclass are the top-level `components` package, next to this one.

One subpackage per process, plus what they share:

- api/: the API process: HTTP routes (api/http/ for what provider routes share),
  the catalog read from the mounted ConfigMap, and the durable scheduler on
  PostgreSQL (api/scheduler/), which calls the analysis runner over HTTP. Raw
  tiles go straight to Ray Serve (`POST /<model-id>/`), not through the API.
- served_model/: the generic served-model host, one Serve app per served model.
- analysis_runner/: the analysis runner, one Serve app for every analysis.
- catalog/, registry/, formats/: catalog files, the scan that finds the
  components they name, and the data formats sinks write; every process loads
  them.
- slides/: the team's slide reading and tissue detection, for analysis sources.
- providers/: where inputs come from and results go (standalone, EMPAIA); they
  plug into both the API and the analysis runner.
- contracts/: data shapes processes hand each other (invocation, result,
  failures, the analysis runner's HTTP format).
"""
