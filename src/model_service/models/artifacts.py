def mlflow(artifact_uri: str) -> str:
    import mlflow.artifacts

    return mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
