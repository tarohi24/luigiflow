import datetime
import pickle
from typing import Dict, NoReturn, Optional, Set
from unittest import TestCase

import luigi
import pandas as pd
import pytest
from luigi import LocalTarget

from luigiflow.savers import save_dataframe, save_pickle
from luigiflow.task import MlflowTask


def test_to_mlflow_tags(monkeypatch):
    class Task(MlflowTask):
        param_int: int = luigi.IntParameter(default=10)
        param_str: str = luigi.Parameter(default="hi")
        param_bool: str = luigi.BoolParameter(default=True)
        param_date: datetime.date = luigi.DateParameter(
            default=datetime.date(2021, 1, 2)
        )
        param_large_value: float = luigi.FloatParameter(default=2e11)
        optional_param: Optional[str] = luigi.Parameter(default=None)

        @classmethod
        def get_tags_to_exclude(cls) -> Set[str]:
            return {"param_int", "param_date", "param_large_value"}

    task = Task()
    TestCase().assertDictEqual(
        task.to_mlflow_tags(),
        {
            "param_str": "hi",
            "param_bool": 1,
        }
    )

    def mock_get_tags_to_exclude(*args, **kwargs):
        return set()

    # overwrite `get_tags_to_exclude`
    monkeypatch.setattr(Task, "get_tags_to_exclude", mock_get_tags_to_exclude)
    TestCase().assertDictEqual(
        task.to_mlflow_tags(),
        {
            "param_int": 10,
            "param_str": "hi",
            "param_bool": 1,
            "param_date": "2021-01-02",
            "param_large_value": 200000000000.0,
        },
    )

    class AnotherTask(Task):
        strange_param = luigi.Parameter(default=Task())

    with pytest.raises(TypeError):
        AnotherTask().to_mlflow_tags()


def test_to_tags_w_parents(monkeypatch):
    class TaskA(MlflowTask):
        param: str = luigi.Parameter(default="hi")

        def requires(self) -> Dict[str, luigi.Task]:
            return dict()

    class TaskB(MlflowTask):
        value: int = luigi.IntParameter(default=1)

        def requires(self) -> Dict[str, luigi.Task]:
            return {"aaa": TaskA()}

    class TaskC(MlflowTask):
        int_param: int = luigi.IntParameter(default=10)

        def requires(self) -> Dict[str, luigi.Task]:
            return dict()

    class MainTask(MlflowTask):
        bool_param: bool = luigi.BoolParameter(default=False)

        def requires(self) -> Dict[str, luigi.Task]:
            return {
                "bbb": TaskB(),
                "ccc": TaskC(),
            }

    assert sorted(MainTask().to_mlflow_tags_w_parent_tags().items()) == sorted({
        "bool_param": 0,
        "ccc.int_param": 10,
        "bbb.value": 1,
        "bbb.aaa.param": "hi",
    }.items())

    # Test non-recurseive

    class MainTaskWoRecursiveTags(MlflowTask):
        bool_param: bool = luigi.BoolParameter(default=True)

        def requires(self) -> Dict[str, luigi.Task]:
            return {
                "bbb": TaskB(),
                "ccc": TaskC(),
            }

        @classmethod
        def output_tags_recursively(cls) -> bool:
            return False

    task = MainTaskWoRecursiveTags()
    TestCase().assertDictEqual(
        task.to_mlflow_tags_w_parent_tags(),
        {
            "bool_param": True,
        },
    )

    # The top task output tags recursively, but one of its dependencies don't
    def mock_return_false(*args, **kwargs):
        return False

    monkeypatch.setattr(TaskB, "output_tags_recursively", mock_return_false)
    task = MainTask()
    TestCase().assertDictEqual(
        task.to_mlflow_tags_w_parent_tags(),
        {
            "bool_param": False,
            "ccc.int_param": 10,
            "bbb.value": 1,
        }
    )


def test_launch_server(artifacts_server):
    class Task(MlflowTask):
        @classmethod
        def get_experiment_name(cls) -> str:
            return "dummy"

        @classmethod
        def get_artifact_filenames(cls) -> Dict[str, str]:
            return {
                "csv": "df.csv",
                "pickle": "df.pickle",
            }

        def requires(self) -> Dict[str, luigi.Task]:
            return dict()

        def _run(self) -> NoReturn:
            df = pd.DataFrame(
                {
                    "a": [0, 1],
                    "b": ["hi", "exmaple"],
                }
            )
            self.save_to_mlflow(
                artifacts_and_save_funcs={
                    "csv": (df, save_dataframe),
                    "pickle": (df, save_pickle),
                }
            )

    task = Task()
    assert task.output() is None
    task.run()
    # Check if the artifacts are saved
    paths: Optional[dict[str, LocalTarget]] = task.output()
    assert paths is not None
    pd.read_csv(paths["csv"].path)
    with open(paths["pickle"].path, "rb") as fin:
        pickle.load(fin)
