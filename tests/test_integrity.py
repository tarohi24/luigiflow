from datetime import datetime
from pathlib import Path
from typing import NoReturn, Protocol, runtime_checkable

import luigi
import pandas as pd
import pytest
from dependency_injector.wiring import inject, Provide
from luigi import LuigiStatusCode

import luigiflow
from luigiflow.config.jsonnet import InvalidJsonnetFileError, JsonnetConfigLoader
from luigiflow.config.run import RunnerConfig
from luigiflow.runner import Runner
from luigiflow.utils.savers import save_dataframe
from luigiflow.task import MlflowTask, TaskConfig
from luigiflow.task_repository import TaskRepository


@runtime_checkable
class SaveCsv(Protocol):

    def save_csv(self, path: Path):
        raise NotImplementedError()


@runtime_checkable
class SaveJson(Protocol):

    def save_json(self, path: Path):
        raise NotImplementedError()


class TaskA(MlflowTask):
    value: float = luigi.FloatParameter()

    config = TaskConfig(
        experiment_name="task",
        protocols=[SaveCsv, ],
    )

    def save_csv(self, path: Path):
        ...

    @classmethod
    def get_artifact_filenames(cls) -> dict[str, str]:
        return {
            "csv": "a.csv",
        }

    @inject
    def requires(self) -> dict[str, luigi.Task]:
        return dict()

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
    )

    def save_csv(self, path: Path):
        ...

    def save_json(self, path: Path):
        ...

    @classmethod
    def get_artifact_filenames(cls) -> dict[str, str]:
        return {
            "csv": "out_b.csv",
            "json": "json.json",
        }

    def requires(self, save_csv_task: type[MlflowTask] = Provide["SaveCsv"]) -> dict[str, luigi.Task]:
        return {
            "a": save_csv_task(),
        }

    def _run(self) -> NoReturn:
        df = pd.DataFrame()
        self.save_to_mlflow(
            artifacts_and_save_funcs={
                "csv": (df, save_dataframe),
                "json": (df, save_dataframe),
            }
        )


def test_run_multiple_tasks(artifacts_server, tmpdir):
    config_path = tmpdir.mkdir("sub").join("config.jsonnet")
    with config_path.open("w") as fout:
        fout.write('''
            {
                "TaskA": {
                    "value": 3.0,
                },
                "TaskB": {
                    "date_start": std.extVar("DATE_START"),
                    "int_value": 1,
                    "message": "Hello!",
                }
            }
            ''')
    runner = Runner(
        config=RunnerConfig(
            mlflow_tracking_uri=artifacts_server.url,
            config_path=config_path,
            use_local_scheduler=True,
            create_experiment_if_not_existing=True,
        ),
        experiment_repository=TaskRepository(
            task_classes=[TaskA, TaskB],
            dependencies={
                "SaveCsv": "TaskA",
                "SaveJson": "TaskB",
            }
        )
    )
    runner.experiment_repository.inject_dependencies(
        module_to_wire=[__name__, ],
    )
    invalid_params = [
        {"DATE": "2021-11-11"},
        {"DATE_START": "2021-11-11"},
    ]
    with pytest.raises(InvalidJsonnetFileError):
        runner.run("SaveJson", params=invalid_params)

    # valid keys, invalid values
    invalid_params = [
        {"DATE_START": "hi!"},  # invalid
        {"DATE_START": "2021-11-11"},  # valid
    ]
    with pytest.raises(ValueError):
        runner.run("SaveJson", params=invalid_params)

    valid_params = [
        {"DATE_START": "2021-11-12"},  # valid
        {"DATE_START": "2021-11-11"},  # valid
    ]
    tasks, res = runner.run("SaveJson", params=valid_params)
    assert len(tasks) == 2
    assert res.status == LuigiStatusCode.SUCCESS
    # Check if all the tasks ran
    for param in valid_params:
        config_loader = JsonnetConfigLoader(external_variables=param)
        with config_loader.load(config_path):
            assert TaskA().complete()


def test_dry_run(artifacts_server):
    config_path = Path(__file__).parent / 'fixture/config.jsonnet'
    tasks, res = luigiflow.run(
        task_cls=TaskB,
        mlflow_tracking_uri=artifacts_server.url,
        config_path=config_path,
        local_scheduler=True,
        create_experiment_if_not_existing=True,
        dry_run=True,
    )
    assert len(tasks) == 1
    assert res is None
