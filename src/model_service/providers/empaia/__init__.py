"""The EMPAIA provider: jobs from EMPAIA's Job Execution Service run as builders.

An EAD (EMPAIA App Description) declares an app's inputs, outputs and modes.

- provider.py: EmpaiaProvider, what EMPAIA adds to the facade and replicas.
- routes.py: the endpoint JES posts jobs to.
- job_adapter.py: admits a job and publishes its results.
- job_mapping.py: maps a job onto a builder (configuration, inputs).
- publishers.py, pixelmap_publisher.py: turn builder outputs into EMPAIA outputs.
- job_inputs.py: fetches a job's input files for a builder replica.
- app_service/: the App Service client (interface, HTTP implementation, data).
- ead.py, ead_builder_io.py: generate each builder's app and its EAD.
- registration.py, credentials.py: register apps in the marketplace.
- settings.py: EMPAIA_* settings.
"""
