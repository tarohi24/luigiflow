import json
from datetime import datetime
from pathlib import Path
from typing import NoReturn

import luigi
import pandas as pd
import pytest
from luigi import LuigiStatusCode

import luigiflow
from luigiflow.config import InvalidJsonnetFileError, JsonnetConfigLoader
from luigiflow.runner import run_multiple_tasks_of_single_task_cls
from luigiflow.savers import save_dataframe
from luigiflow.task import MlflowTask


class TaskA(MlflowTask):
    date_start: datetime.date = luigi.DateParameter()

    @classmethod
    def get_experiment_name(cls) -> str:
        return "task_a"

    @classmethod
    def get_artifact_filenames(cls) -> dict[str, str]:
        return {
            "out": "a.csv",
        }

    def requires(self) -> dict[str, luigi.Task]:
        return dict()

    def _run(self) -> NoReturn:
        df = pd.DataFrame()
        self.save_to_mlflow(
            artifacts_and_save_funcs={
                "out": (df, save_dataframe),
            }
        )


class TaskB(MlflowTask):
    value: float = luigi.FloatParameter()
    int_value: int = luigi.IntParameter()
    message: str = luigi.Parameter()

    @classmethod
    def get_experiment_name(cls) -> str:
        return "b"

    @classmethod
    def get_artifact_filenames(cls) -> dict[str, str]:
        return {
            "out_b": "out_b.txt",
        }

    def requires(self) -> dict[str, luigi.Task]:
        return {
            "a": TaskA(),
        }

    def _run(self) -> NoReturn:
        df = pd.DataFrame()
        self.save_to_mlflow(
            artifacts_and_save_funcs={
                "out_b": (df, save_dataframe),
            }
        )


def test_run(artifacts_server):
    config_path = Path(__file__).parent / 'fixture/config.jsonnet'
    tasks, res = luigiflow.run(
        task_cls=TaskB,
        mlflow_tracking_uri=artifacts_server.url,
        config_path=config_path,
        local_scheduler=True,
        create_experiment_if_not_existing=True,
    )
    assert len(tasks) == 1
    assert res.status == LuigiStatusCode.SUCCESS


def test_run_multiple_tasks(artifacts_server, tmpdir):
    config_path = tmpdir.mkdir("sub").join("config.jsonnet")
    with config_path.open("w") as fout:
        fout.write('''
            {
                "TaskA": {
                    "date_start": std.extVar("DATE_START"),
                }
            }
            ''')
    kwargs = dict(
        mlflow_tracking_uri=artifacts_server.url,
        config_path=config_path,
        local_scheduler=True,
        create_experiment_if_not_existing=True,
    )
    invalid_params = [
        {"DATE": "2021-11-11"},
        {"DATE_START": "2021-11-11"},
    ]
    with pytest.raises(InvalidJsonnetFileError):
        run_multiple_tasks_of_single_task_cls(
            task_cls=TaskA,
            params=invalid_params,
            **kwargs
        )
    # valid keys, invalid values
    invalid_params = [
        {"DATE_START": "hi!"},  # invalid
        {"DATE_START": "2021-11-11"},  # valid
    ]
    with pytest.raises(ValueError):
        run_multiple_tasks_of_single_task_cls(
            task_cls=TaskA,
            params=invalid_params,
            **kwargs
        )
    valid_params = [
        {"DATE_START": "2021-11-12"},  # valid
        {"DATE_START": "2021-11-11"},  # valid
    ]
    tasks, res = run_multiple_tasks_of_single_task_cls(
        task_cls=TaskA,
        params=valid_params,
        **kwargs
    )
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
