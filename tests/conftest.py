import os
import tempfile
from pathlib import Path

import mlflow
import pytest

from luigiflow.testing import get_safe_port, launch_mlflow_server, ArtifactsServer

LOCALHOST = "127.0.0.1"


@pytest.fixture(scope="module")
def artifacts_server():
    with tempfile.TemporaryDirectory() as tmpdir:
        port = get_safe_port()
        backend_store_uri = os.path.join(tmpdir, "mlruns")
        artifacts_destination = os.path.join(tmpdir, "mlartifacts")
        process = launch_mlflow_server(
            host=LOCALHOST,
            port=port,
            backend_store_uri=backend_store_uri,
            default_artifact_root=artifacts_destination,
        )
        tracking_url = f"http://{LOCALHOST}:{port}"
        mlflow.set_tracking_uri(tracking_url)
        yield ArtifactsServer(
            backend_store_uri,
            artifacts_destination,
            tracking_url,
            process,
        )
        process.kill()


@pytest.fixture()
def project_root() -> Path:
    return Path(__file__).parent.parent
