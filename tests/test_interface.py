from datetime import datetime
from pathlib import Path
from typing import NoReturn, Dict

import luigi
import pandas as pd
from luigi import LuigiStatusCode

import luigiflow
from luigiflow.savers import save_dataframe
from luigiflow.task import MlflowTask


class TaskA(MlflowTask):
    date_start: datetime.date = luigi.DateParameter()

    @classmethod
    def get_experiment_name(cls) -> str:
        return "task_a"

    @classmethod
    def get_artifact_filenames(cls) -> Dict[str, str]:
        return {
            "out": "a.csv",
        }

    def requires(self) -> Dict[str, luigi.Task]:
        return dict()

    def _run(self) -> NoReturn:
        df = pd.DataFrame()
        self.save_to_mlflow(
            artifacts_and_save_funcs={
                "out": (df, save_dataframe),
            }
        )
        pass


class TaskB(MlflowTask):
    value: float = luigi.FloatParameter()
    int_value: int = luigi.IntParameter()
    message: str = luigi.Parameter()

    @classmethod
    def get_experiment_name(cls) -> str:
        return "b"

    @classmethod
    def get_artifact_filenames(cls) -> Dict[str, str]:
        return {
            "out_b": "out_b.txt",
        }

    def requires(self) -> Dict[str, luigi.Task]:
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
    res = luigiflow.run(
        task_cls=TaskB,
        mlflow_tracking_uri=artifacts_server.url,
        config_path=config_path,
        local_scheduler=True,
        create_experiment_if_not_existing=True,
    )
    assert res.status == LuigiStatusCode.SUCCESS
