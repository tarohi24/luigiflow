import mlflow


def initialize(
    mlflow_tracking_uri: str,
):
    """
    Common procedure for every flow
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
