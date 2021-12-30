from pathlib import Path
from typing import NoReturn, Dict

import luigi
import pandas as pd
from luigi.configuration import get_config

from luigiflow.config import JsonnetConfigLoader
from luigiflow.savers import save_dataframe
from luigiflow.task import MlflowTask


class SomeTask(MlflowTask):
    int_param: int = luigi.IntParameter()

    @classmethod
    def get_experiment_name(cls) -> str:
        return "example"

    @classmethod
    def get_artifact_filenames(cls) -> Dict[str, str]:
        return {
            "data": "data.csv",
        }

    def requires(self) -> Dict[str, luigi.Task]:
        return dict()

    def _run(self) -> NoReturn:
        df = pd.DataFrame([
            {"label": "A", "score": 1.1},
            {"label": "B", "score": 0.2},
        ])
        self.save_to_mlflow(
            artifacts_and_save_funcs={
                "data": (df, save_dataframe),
            }
        )


def test_load_config(project_root, tmpdir):
    tmpfile = tmpdir.mkdir("sub").join("config.jsonnet")
    with tmpfile.open("w") as fout:
        fout.write("""
        {
            "SomeTask": {
                "int_param": 3,
            },
        }
        """)
    loader = JsonnetConfigLoader()
    with loader.load(tmpfile):
        task = SomeTask()
        assert task.int_param == 3

