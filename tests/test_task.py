import datetime
import os
import tempfile
from abc import ABC
from typing import NoReturn, Dict
from unittest import TestCase

import luigi
import pytest

from luigiflow.task import MlflowTask, MlflowTagValue
from luigiflow.testing import (
    ArtifactsServer,
    get_safe_port,
    is_windows,
    launch_mlflow_server,
)

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
            backend_store_uri,
            default_artifact_root,
            artifacts_destination,
            url,
            process,
        )
        process.kill()


def test_to_mlflow_tags():
    class Task(MlflowTask):
        param_int: int = luigi.IntParameter(default=10)
        param_str: str = luigi.Parameter(default="hi")
        param_bool: str = luigi.BoolParameter(default=True)
        param_date: datetime.date = luigi.DateParameter(default=datetime.date(2021, 1, 2))

    task = Task()
    TestCase().assertDictEqual(task.to_mlflow_tags(), {
        "param_int": 10,
        "param_str": "hi",
        "param_bool": 1,
        "param_date": "2021-01-02",
    })

    class AnotherTask(Task):
        strange_param = luigi.Parameter(default=Task())

    with pytest.raises(TypeError):
        AnotherTask().to_mlflow_tags()


def test_to_tags():

    class DummyTaskBase(MlflowTask, ABC):

        @classmethod
        def get_experiment_name(cls) -> str:
            return 'dummy'

        @classmethod
        def get_artifact_filenames(cls) -> Dict[str, str]:
            return dict()

        def _run(self) -> NoReturn:
            self.get_params()
            pass

    class TaskA(DummyTaskBase):

        def requires(self):
            return dict()

        def to_mlflow_tags(self):
            return {"param_1": 10}

    class TaskB(MlflowTask):
        message: str = luigi.Parameter(default='hi')
        def requires(self): return dict()

        def to_mlflow_tags(self): return {"message": self.message}

        def _run(self): ...

    class TaskC(MlflowTask):
        def requires(self): return {"bbb": TaskB()}

        def to_mlflow_tags(self): return dict()

        def _run(self): ...

    class TaskD(MlflowTask):
        threshold: float = luigi.FloatParameter(default=2e+3)

        def requires(self): return {'aaa': TaskA(), 'ccc': TaskC()}

        def to_mlflow_tags(self): return {"threshold": self.threshold}

        def _run(self): ...

    task = TaskD()
    task.to_mlflow_tags_w_parent_tags()
