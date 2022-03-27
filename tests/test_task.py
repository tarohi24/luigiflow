import datetime
import pickle
from typing import NoReturn, Optional, Protocol, runtime_checkable
from unittest import TestCase

import luigi
import pandas as pd
import pytest
from luigi import LocalTarget

from luigiflow.task import MlflowTask, TaskConfig, TryingToSaveUndefinedArtifact, MlflowTaskProtocol
from luigiflow.utils.savers import save_dataframe, save_pickle, save_json


def test_task_protocol_and_implementation_consistent():
    assert issubclass(MlflowTask, MlflowTaskProtocol)


def test_to_mlflow_tags(monkeypatch):

    @runtime_checkable
    class DummyProtocol(MlflowTaskProtocol, Protocol):

        def do_nothing(self):
            raise NotImplementedError()

    class Task(MlflowTask):
        param_int: int = luigi.IntParameter(default=10)
        param_str: str = luigi.Parameter(default="hi")
        param_bool: str = luigi.BoolParameter(default=True)
        param_date: datetime.date = luigi.DateParameter(
            default=datetime.date(2021, 1, 2)
        )
        param_large_value: float = luigi.FloatParameter(default=2e11)
        optional_param: Optional[str] = luigi.Parameter(default=None)
        config = TaskConfig(
            experiment_name="task",
            protocols=[DummyProtocol, ],
            tags_to_exclude={"param_int", "param_date", "param_large_value"},
        )

        def do_nothing(self):
            ...

    task = Task()
    TestCase().assertDictEqual(
        task.to_mlflow_tags(),
        {
            "param_str": "hi",
            "param_bool": 1,
        }
    )

    # disable `tags_to_exclude`
    monkeypatch.setattr(Task, "tags_to_exclude", set())
    expected = {
        "param_int": 10,
        "param_str": "hi",
        "param_bool": 1,
        "param_date": "2021-01-02",
        "param_large_value": 200000000000.0,
    }
    assert task.to_mlflow_tags() == expected

    class AnotherTask(Task):
        strange_param = luigi.Parameter(default=Task())  # invalid value
        config = TaskConfig(
            experiment_name="dummy",
            protocols=[],
            artifact_filenames=dict(),
        )

    with pytest.raises(TypeError):
        AnotherTask().to_mlflow_tags()


def test_to_tags_w_parents(monkeypatch):

    class TaskA(MlflowTask):
        param: str = luigi.Parameter(default="hi")
        config = TaskConfig(
            experiment_name="dummy",
            protocols=[],
        )

        def _run(self) -> NoReturn:
            ...

    class TaskB(MlflowTask):
        value: int = luigi.IntParameter(default=1)
        config = TaskConfig(
            experiment_name="dummy",
            protocols=[],
        )

        def requires(self) -> dict[str, MlflowTaskProtocol]:
            return {
                "aaa": TaskA(),
            }

        def _run(self) -> NoReturn:
            ...

    class TaskC(MlflowTask):
        int_param: int = luigi.IntParameter(default=10)
        config = TaskConfig(
            experiment_name="dummy",
            protocols=[],
        )

        def _run(self) -> NoReturn:
            ...

    class MainTask(MlflowTask):
        bool_param: bool = luigi.BoolParameter(default=False)
        config = TaskConfig(
            experiment_name="dummy",
            protocols=[],
        )

        def requires(self) -> dict[str, MlflowTaskProtocol]:
            return {
                "bbb": TaskB(),
                "ccc": TaskC(),
            }

        def _run(self) -> NoReturn:
            ...

    assert sorted(MainTask().to_mlflow_tags_w_parent_tags().items()) == sorted({
        "bool_param": 0,
        "ccc.int_param": 10,
        "bbb.value": 1,
        "bbb.aaa.param": "hi",
    }.items())

    # Test non-recursive outputs
    class MainTaskWoRecursiveTags(MlflowTask):
        bool_param: bool = luigi.BoolParameter(default=True)
        config = TaskConfig(
            experiment_name="dummy",
            protocols=[],
            output_tags_recursively=False,
        )

        def requires(self) -> dict[str, MlflowTaskProtocol]:
            return {
                "bbb": TaskB(),
                "ccc": TaskC(),
            }

        def _run(self) -> NoReturn:
            ...

    task = MainTaskWoRecursiveTags()
    TestCase().assertDictEqual(
        task.to_mlflow_tags_w_parent_tags(),
        {
            "bool_param": True,
        },
    )

    monkeypatch.setattr(TaskB, "output_tags_recursively", False)
    task = MainTask()
    TestCase().assertDictEqual(
        task.to_mlflow_tags_w_parent_tags(),
        {
            "bool_param": False,
            "ccc.int_param": 10,
            "bbb.value": 1,
        }
    )


def test_save_artifacts(artifacts_server):
    class Task(MlflowTask):
        config = TaskConfig(
            experiment_name="dummy",
            artifact_filenames={
                "csv": "df.csv",
                "pickle": "df.pickle",
            },
            protocols=[],
        )

        def _run(self) -> NoReturn:
            df = pd.DataFrame(
                {
                    "a": [0, 1],
                    "b": ["hi", "example"],
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


def test_save_artifacts_but_files_are_mismatched(artifacts_server):
    class InvalidTask(MlflowTask):
        config = TaskConfig(
            experiment_name="dummy",
            protocols=[],
            artifact_filenames={
                "csv": "csv.csv",
            }
        )

        def _run(self) -> NoReturn:
            self.save_to_mlflow(
                artifacts_and_save_funcs={
                    "json": (dict(), save_json),
                }
            )

    task = InvalidTask()
    assert task.output() is None
    with pytest.raises(TryingToSaveUndefinedArtifact):
        task.run()
