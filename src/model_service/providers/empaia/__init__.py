"""The EMPAIA provider: jobs from EMPAIA's Job Execution Service run as builders.

An EAD (EMPAIA App Description) declares an app's inputs, outputs and modes.

- provider.py: EmpaiaProvider, what EMPAIA adds to the facade and replicas.
- routes.py: the endpoint JES posts jobs to.
- scheduler_adapter.py: admits a job and publishes its results.
- publishers.py, pixelmaps.py: turn builder outputs into EMPAIA outputs.
- input_access.py: fetches a job's input files for a builder replica.
- gateway.py: the App Service interface; http_gateway.py implements it.
- app_service_types.py, responses.py: data the App Service returns.
- ead.py, ead_io.py: generate each builder's app and its EAD.
- registration.py, credentials.py: register apps in the marketplace.
- settings.py: EMPAIA_* settings.
"""
