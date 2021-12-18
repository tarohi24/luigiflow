import os
import tempfile

import pytest

from luigiflow.testing import get_safe_port, is_windows, launch_mlflow_server, ArtifactsServer

LOCALHOST = "127.0.0.1"


@pytest.fixture(scope="module")
def artifacts_server():
    with tempfile.TemporaryDirectory() as tmpdir:
        port = get_safe_port()
        backend_store_uri = os.path.join(tmpdir, "mlruns")
        artifacts_destination = os.path.join(tmpdir, "mlartifacts")
        url = f"http://{LOCALHOST}:{port}"
        default_artifact_root = f"{url}/api/2.0/mlflow-artifacts/artifacts"
        uri_prefix = "file:///" if is_windows() else ""
        process = launch_mlflow_server(
            LOCALHOST,
            port,
            uri_prefix + backend_store_uri,
            default_artifact_root,
            uri_prefix + artifacts_destination,
        )
        yield ArtifactsServer(
            backend_store_uri, default_artifact_root, artifacts_destination, url, process
        )
        process.kill()


def test_a(artifacts_server):
    pass