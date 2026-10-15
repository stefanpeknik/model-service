"""Download model artifacts for model deployment construction.

This model loader is separate from model_service.scheduler.providers, whose adapters connect
workflows to caller data and external job lifecycles.
"""


def mlflow(artifact_uri: str) -> str:
    """Download an MLflow artifact URI and return its local filesystem path; download failures propagate to deployment construction.

    Args:
        artifact_uri: MLflow artifact location to download.

    Returns:
        str: Local filesystem path of the downloaded MLflow artifact.

    Raises:
        Exception: MLflow cannot resolve or download the artifact; its original error
            propagates.
    """
    import mlflow.artifacts

    return mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
