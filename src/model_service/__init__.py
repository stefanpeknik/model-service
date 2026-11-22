"""RationAI model service: runs pathology models on Ray Serve behind one API.

One top-level package per process, plus what they share:

- api/: the API process: HTTP routes (api/http/ for what provider routes share),
  the catalog read from the mounted ConfigMap, and the durable scheduler on
  PostgreSQL (api/scheduler/), which calls the analysis runner over HTTP. Raw
  tiles go straight to Ray Serve (`POST /<model-id>/`), not through the API.
- served_model/: the generic served-model host, one Serve app per served model.
- analysis_runner/: the analysis runner, one Serve app for every analysis.
- catalog/, components/, formats/: catalog files, the components they name,
  and the data formats sinks write; every process loads them.
- slides/: the team's slide reading and tissue detection, for analysis sources.
- providers/: where inputs come from and results go (standalone, EMPAIA); they
  plug into both the API and the analysis runner.
- contracts/: data shapes processes hand each other (invocation, result,
  failures, the analysis runner's HTTP format).
"""
