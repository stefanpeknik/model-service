"""The EMPAIA provider: jobs from EMPAIA's Job Execution Service run analyses.

An EAD (EMPAIA App Description) declares an app's inputs, outputs and modes.

- provider.py: EmpaiaProvider, what EMPAIA adds to the control plane and the runner.
- jobs/: an EMPAIA job run as an analysis: routes.py is where EMPAIA's Job
  Execution Service (JES) posts it, payload.py parses that post, adapter.py
  admits the job and publishes its results, gateways.py opens the App Service
  connection for each, mapping.py maps the job's configuration and inputs,
  inputs.py makes its input files with the converters admission recorded.
- app_service/: the App Service gateway, one method per SDK call family
  (interface, HTTP implementation, data).
- ead/: apps.py generates each exposed analysis's app and EAD from its
  converters' and publishers' fragments; vocabulary.py holds modes, triggers
  and viewer settings.
- Converters and publishers are components: src/components/providers/empaia/.
- registration.py, credentials.py: register apps in the marketplace.
- settings.py: EMPAIA_* settings.
"""
