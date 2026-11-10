"""The EMPAIA provider: jobs from EMPAIA's Job Execution Service run as builders.

An EAD (EMPAIA App Description) declares an app's inputs, outputs and modes.

- provider.py: EmpaiaProvider, what EMPAIA adds to the facade and replicas.
- routes.py: the endpoint JES posts jobs to.
- jobs/: an EMPAIA job run as a builder: adapter.py admits it and publishes its
  results, mapping.py maps it onto a builder, inputs.py fetches its input files.
- publishing/: publishers.py turns builder outputs into EMPAIA outputs;
  pixelmap.py publishes score grids as pixelmaps.
- app_service/: the App Service client (interface, HTTP implementation, data).
- ead/: apps.py generates each builder's app and its EAD; builder_io.py holds
  the builders' EAD io entries.
- registration.py, credentials.py: register apps in the marketplace.
- settings.py: EMPAIA_* settings.
"""
