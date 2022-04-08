from datetime import datetime
from pathlib import Path
from typing import NoReturn, Protocol, runtime_checkable, TypedDict

import luigi
import pandas as pd
import pytest
from luigi import LuigiStatusCode

from luigiflow.config.run import RunnerConfig
from luigiflow.runner import Runner
from luigiflow.task import MlflowTask, TaskConfig, MlflowTaskProtocol
from luigiflow.task_repository import TaskRepository
from luigiflow.utils.savers import save_dataframe


@runtime_checkable
class SaveCsv(MlflowTaskProtocol, Protocol):
    def save_csv(self, path: Path):
        raise NotImplementedError()

    def get_value(self) -> float:
        raise NotImplementedError


@runtime_checkable
class SaveJson(MlflowTaskProtocol, Protocol):
    def save_json(self, path: Path):
        raise NotImplementedError()


class TaskA(MlflowTask):
    value: float = luigi.FloatParameter()

    config = TaskConfig(
        experiment_name="task",
        protocols=[
            SaveCsv,
        ],
        requirements=dict(),
        artifact_filenames={
            "csv": "a.csv",
        },
    )

    def get_value(self) -> float:
        return self.value

    def save_csv(self, path: Path):
        ...

    def _run(self) -> NoReturn:
        df = pd.DataFrame()
        self.save_to_mlflow(
            artifacts_and_save_funcs={
                "csv": (df, save_dataframe),
            }
        )


class Requirements(TypedDict):
    a: SaveCsv


class TaskB(MlflowTask[Requirements]):
    date_start: datetime.date = luigi.DateParameter()
    int_value: int = luigi.IntParameter()
    message: str = luigi.Parameter()
    config = TaskConfig(
        experiment_name="task",
        protocols=[SaveCsv, SaveJson],
        requirements={
            "a": SaveCsv,
        },
        artifact_filenames={
            "csv": "out_b.csv",
            "json": "json.json",
        },
    )

    def get_value(self):
        ...

    def save_csv(self, path: Path):
        ...

    def save_json(self, path: Path):
        ...

    def _run(self) -> NoReturn:
        df = pd.DataFrame()
        self.save_to_mlflow(
            artifacts_and_save_funcs={
                "csv": (df, save_dataframe),
                "json": (df, save_dataframe),
            }
        )


@pytest.fixture()
def runner(artifacts_server) -> Runner:
    runner = Runner(
        config=RunnerConfig(
            mlflow_tracking_uri=artifacts_server.url,
            use_local_scheduler=True,
            create_experiment_if_not_existing=True,
        ),
        experiment_repository=TaskRepository(
            task_classes=[TaskA, TaskB],
        ),
    )
    return runner


@pytest.fixture()
def sample_task_param_path(tmpdir) -> Path:
    config_path = tmpdir.mkdir("sub").join("config.jsonnet")
    with config_path.open("w") as fout:
        fout.write(
            """
                local val = 3.0;
                {
                    cls: "TaskB",
                    params: {
                        date_start: "2011-11-10",
                        int_value: 1,
                        message: "Hello!",
                    },
                    requires: {
                        a: {
                            cls: "TaskA",
                            params: {
                                value: val,
                            },
                        }
                    }
                }
                """
        )
    return config_path


def test_run_with_single_param(runner, sample_task_param_path):
    task, res = runner.run("SaveJson", sample_task_param_path)
    assert isinstance(task, TaskB)
    assert res.status == LuigiStatusCode.SUCCESS
    assert task.int_value == 1
    assert task.requires()["a"].get_value() == 3.0


def test_dry_run(runner, sample_task_param_path):
    task, res = runner.run(
        protocol_name="SaveJson",
        config_jsonnet_path=sample_task_param_path,
        dry_run=True,
    )
    assert isinstance(task, MlflowTask)
    assert res is None
