"""
These are copied from the MLflow library.
The original source codes are served under the Apache 2.0 license.
Source:
- https://github.com/mlflow/mlflow/blob/fb2972fd0d4e6eb9c8d6050d75a6f6c1c56427a5/tests/tracking/test_mlflow_artifacts.py
- https://github.com/mlflow/mlflow/blob/2b67fc156c19bf3c9aa8b044b0800a2696cb486a/tests/tracking/integration_test_utils.py#L17
"""
import os
import socket
import subprocess
import time
from collections import namedtuple

ArtifactsServer = namedtuple(
    "ArtifactsServer",
    ["backend_store_uri", "default_artifact_root", "artifacts_destination", "url", "process"],
)

LOCALHOST = "127.0.0.1"


def get_safe_port():
    """Returns an ephemeral port that is very likely to be free to bind to."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((LOCALHOST, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def is_windows():
    return os.name == "nt"


def _await_server_up_or_die(port, timeout=60):
    """Waits until the local flask server is listening on the given port."""
    print("Awaiting server to be up on %s:%s" % (LOCALHOST, port))
    start_time = time.time()
    connected = False
    while not connected and time.time() - start_time < timeout:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((LOCALHOST, port))
        if result == 0:
            connected = True
        else:
            print("Server not yet up, waiting...")
            time.sleep(0.5)
    if not connected:
        raise Exception("Failed to connect on %s:%s after %s seconds" % (LOCALHOST, port, timeout))
    print("Server is up on %s:%s!" % (LOCALHOST, port))


# The original name is `_launch_server`
def launch_mlflow_server(host, port, backend_store_uri, default_artifact_root, artifacts_destination):
    extra_cmd = [] if is_windows() else ["--gunicorn-opts", "--log-level debug"]
    cmd = [
        "mlflow",
        "server",
        "--host",
        host,
        "--port",
        str(port),
        "--backend-store-uri",
        backend_store_uri,
        # MODIFICATION: removed the `save_artifacts` flag
        "--default-artifact-root",
        default_artifact_root,
        "--artifacts-destination",
        artifacts_destination,
        *extra_cmd,
    ]
    process = subprocess.Popen(cmd)
    _await_server_up_or_die(port)
    return process
