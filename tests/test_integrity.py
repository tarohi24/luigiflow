from datetime import datetime
from pathlib import Path
from typing import NoReturn, Protocol, runtime_checkable

import luigi
import pandas as pd
import pytest
from dependency_injector.wiring import Provide
from luigi import LuigiStatusCode

from luigiflow.config.jsonnet import InvalidJsonnetFileError, JsonnetConfigLoader
from luigiflow.config.run import RunnerConfig
from luigiflow.runner import Runner
from luigiflow.task import MlflowTask, TaskConfig, MlflowTaskProtocol
from luigiflow.task_repository import TaskRepository
from luigiflow.utils.savers import save_dataframe


@runtime_checkable
class SaveCsv(MlflowTaskProtocol, Protocol):

    def save_csv(self, path: Path):
        raise NotImplementedError()


@runtime_checkable
class SaveJson(MlflowTaskProtocol, Protocol):

    def save_json(self, path: Path):
        raise NotImplementedError()


class TaskA(MlflowTask):
    value: float = luigi.FloatParameter()

    config = TaskConfig(
        experiment_name="task",
        protocols=[SaveCsv, ],
        requirements=dict(),
        artifact_filenames={
            "csv": "a.csv",
        }
    )

    def save_csv(self, path: Path):
        ...

    def _run(self) -> NoReturn:
        df = pd.DataFrame()
        self.save_to_mlflow(
            artifacts_and_save_funcs={
                "csv": (df, save_dataframe),
            }
        )


class TaskB(MlflowTask):
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
        }
    )

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
        )
    )
    return runner


def test_run_with_single_param(runner, tmpdir):
    config_path = tmpdir.mkdir("sub").join("config.jsonnet")
    with config_path.open("w") as fout:
        fout.write('''
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
            ''')
    runner.run("SaveJson", config_path)



def test_run_multiple_tasks(runner, tmpdir):
    config_path = tmpdir.mkdir("sub").join("config.jsonnet")
    with config_path.open("w") as fout:
        fout.write('''
            local val = 3.0;
            {
                type: "TaskB",
                params: {
                    date_start: std.extVar("DATE_START"),
                    int_value: 1,
                    message: "Hello!",
                },
                requires: {
                    a: {
                        type: "TaskA",
                        params: {
                            value: val,
                        },
                    }
                }
            }
            ''')
    invalid_params = [
        {"DATE": "2021-11-11"},
        {"DATE_START": "2021-11-11"},
    ]
    with pytest.raises(InvalidJsonnetFileError):
        runner.run("SaveJson", config_path)

    # valid keys, invalid values
    invalid_params = [
        {"DATE_START": "hi!"},  # invalid
        {"DATE_START": "2021-11-11"},  # valid
    ]
    with pytest.raises(ValueError):
        runner.run("SaveJson", config_path)

    valid_params = [
        {"DATE_START": "2021-11-12"},  # valid
        {"DATE_START": "2021-11-11"},  # valid
    ]
    task, res = runner.run("SaveJson", config_path)
    assert len(task) == 2
    assert res.status == LuigiStatusCode.SUCCESS
    # Check if all the tasks ran
    for param in valid_params:
        config_loader = JsonnetConfigLoader(external_variables=param)
        with config_loader.load(runner.config.config_path):
            assert TaskA().complete()


def test_dry_run(runner):
    tasks, res = runner.run(
        protocol_name="SaveJson",
        external_params=[
            {"DATE_START": "2021-11-11"},
        ],
        dry_run=True,
    )
    assert len(tasks) == 1
    assert res is None
