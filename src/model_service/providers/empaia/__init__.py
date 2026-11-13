"""The EMPAIA provider: jobs from EMPAIA's Job Execution Service run as builders.

An EAD (EMPAIA App Description) declares an app's inputs, outputs and modes.

- provider.py: EmpaiaProvider, what EMPAIA adds to the API and replicas.
- jobs/: an EMPAIA job run as a builder: routes.py is where EMPAIA's Job
  Execution Service (JES) posts it, payload.py parses that post, adapter.py
  admits the job and publishes its results, gateways.py opens the App Service
  connection for each, mapping.py maps the job onto a builder, inputs.py fetches
  its input files.
- publishing/: publishers.py turns builder outputs into EMPAIA outputs.
- app_service/: the App Service client (interface, HTTP implementation, data).
- ead/: builder_io.py holds the builders' EAD io entries.
- settings.py: EMPAIA_* settings.
"""
