import datetime
import pickle
from typing import NoReturn, Optional
from unittest import TestCase

import luigi
import pandas as pd
import pytest
from luigi import LocalTarget

from luigiflow.task import (
    MlflowTask,
    TaskConfig,
    TryingToSaveUndefinedArtifact,
    MlflowTaskProtocol,
)
from luigiflow.task_repository import TaskRepository
from luigiflow.utils.savers import save_dataframe, save_pickle, save_json


class DummyProtocol(MlflowTaskProtocol):
    ...


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
        config = TaskConfig(
            protocols=[
                DummyProtocol,
            ],
            requirements=dict(),
            tags_to_exclude={"param_int", "param_date", "param_large_value"},
        )

        def do_nothing(self):
            ...

    repo = TaskRepository(
        [
            Task,
        ]
    )
    task = repo.generate_task_tree(
        task_params={
            "cls": "Task",
            "params": dict(),
        },
        protocol="DummyProtocol",
    )
    TestCase().assertDictEqual(
        task.to_mlflow_tags(),
        {
            "name": "Task",
            "param_str": "hi",
            "param_bool": 1,
        },
    )

    # disable `tags_to_exclude`
    monkeypatch.setattr(Task, "tags_to_exclude", set())
    expected = {
        "name": "Task",
        "param_int": 10,
        "param_str": "hi",
        "param_bool": 1,
        "param_date": "2021-01-02",
        "param_large_value": 200000000000.0,
    }
    assert task.to_mlflow_tags() == expected

    class AnotherTask(MlflowTask):
        strange_param = luigi.Parameter(default=Task)  # invalid value
        config = TaskConfig(
            protocols=[
                DummyProtocol,
            ],
            requirements=dict(),
            artifact_filenames=dict(),
        )

    repo._protocols["DummyProtocol"].register(AnotherTask)
    task = repo.generate_task_tree(
        task_params={
            "cls": "AnotherTask",
            "params": dict(),
        },
        protocol="DummyProtocol",
    )

    with pytest.raises(TypeError):
        task.to_mlflow_tags()


def test_to_tags_w_parents(monkeypatch):
    class ITaskA(MlflowTaskProtocol):
        ...

    class ITaskB(MlflowTaskProtocol):
        ...

    class ITaskC(MlflowTaskProtocol):
        ...

    class IMainTask(MlflowTaskProtocol):
        ...

    class TaskA(MlflowTask[dict]):
        param: str = luigi.Parameter(default="hi")
        config = TaskConfig(
            protocols=[
                ITaskA,
            ],
            requirements=dict(),
        )

        def _run(self) -> NoReturn:
            ...

    class TaskB(MlflowTask[dict]):
        value: int = luigi.IntParameter(default=1)
        config = TaskConfig(
            protocols=[
                ITaskB,
            ],
            requirements={
                "aaa": ITaskA,
            },
        )

        def _run(self) -> NoReturn:
            ...

    class TaskC(MlflowTask[dict]):
        int_param: int = luigi.IntParameter(default=10)
        config = TaskConfig(
            protocols=[
                ITaskC,
            ],
            requirements=dict(),
        )

        def _run(self) -> NoReturn:
            ...

    class MainTask(MlflowTask[dict]):
        bool_param: bool = luigi.BoolParameter(default=False)
        config = TaskConfig(
            protocols=[
                IMainTask,
            ],
            requirements={
                "bbb": ITaskB,
                "ccc": ITaskC,
            },
        )

        def _run(self) -> NoReturn:
            ...

    task_repo = TaskRepository(
        task_classes=[TaskA, TaskB, TaskC, MainTask],
    )
    task_params = {
        "cls": "MainTask",
        "params": {},
        "requires": {
            "bbb": {
                "cls": "TaskB",
                "params": {},
                "requires": {
                    "aaa": {
                        "cls": "TaskA",
                    }
                },
            },
            "ccc": {
                "cls": "TaskC",
            },
        },
    }
    main_task = task_repo.generate_task_tree(
        task_params=task_params,
        protocol="IMainTask",
    )

    assert sorted(main_task.to_mlflow_tags_w_parent_tags().items()) == sorted(
        {
            "name": "MainTask",
            "bool_param": 0,
            "ccc.name": "TaskC",
            "ccc.int_param": 10,
            "bbb.name": "TaskB",
            "bbb.value": 1,
            "bbb.aaa.name": "TaskA",
            "bbb.aaa.param": "hi",
        }.items()
    )

    # Test non-recursive outputs
    class MainTaskWoRecursiveTags(MlflowTask):
        bool_param: bool = luigi.BoolParameter(default=True)
        config = TaskConfig(
            protocols=[
                IMainTask,
            ],
            output_tags_recursively=False,
            requirements={
                "bbb": ITaskB,
                "ccc": ITaskC,
            },
        )

        def _run(self) -> NoReturn:
            ...

    task_repo._protocols["IMainTask"].register(MainTaskWoRecursiveTags)
    task_params["cls"] = "MainTaskWoRecursiveTags"
    task = task_repo.generate_task_tree(
        task_params=task_params,
        protocol="IMainTask",
    )
    TestCase().assertDictEqual(
        task.to_mlflow_tags_w_parent_tags(),
        {
            "name": "MainTaskWoRecursiveTags",
            "bool_param": True,
        },
    )

    monkeypatch.setattr(TaskB, "output_tags_recursively", False)
    task_params["cls"] = "MainTask"
    task = task_repo.generate_task_tree(
        task_params=task_params,
        protocol="IMainTask",
    )
    TestCase().assertDictEqual(
        task.to_mlflow_tags_w_parent_tags(),
        {
            "name": "MainTask",
            "bool_param": False,
            "ccc.name": "TaskC",
            "ccc.int_param": 10,
            "bbb.value": 1,
            "bbb.name": "TaskB",
        },
    )


def test_save_artifacts(artifacts_server):
    def dummy_fn(path: str):
        with open(path, "w") as fout:
            fout.write("hello!\n")

    class Task(MlflowTask):
        config = TaskConfig(
            artifact_filenames={
                "csv": "df.csv",
                "pickle": "df.pickle",
                "text": "text.txt",
            },
            protocols=[
                DummyProtocol,
            ],
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
                    "text": dummy_fn,
                }
            )

    task = TaskRepository([Task,]).generate_task_tree(
        task_params={
            "cls": "Task",
        },
        protocol="DummyProtocol",
    )
    assert task.output() is None
    task.run()
    # Check if the artifacts are saved
    paths: Optional[dict[str, LocalTarget]] = task.output()
    assert paths is not None
    pd.read_csv(paths["csv"].path)
    with open(paths["pickle"].path, "rb") as fin:
        pickle.load(fin)
    with open(paths["text"].path) as fin:
        text = fin.read()
    assert text == "hello!\n"


def test_save_artifacts_but_files_are_mismatched(artifacts_server):
    class InvalidTask(MlflowTask):
        config = TaskConfig(
            protocols=[
                DummyProtocol,
            ],
            artifact_filenames={
                "csv": "csv.csv",
            },
        )

        def _run(self) -> NoReturn:
            self.save_to_mlflow(
                artifacts_and_save_funcs={
                    "json": (dict(), save_json),
                }
            )

    task = TaskRepository([InvalidTask,]).generate_task_tree(
        task_params={"cls": "InvalidTask"},
        protocol=DummyProtocol,
    )
    assert task.output() is None
    with pytest.raises(TryingToSaveUndefinedArtifact):
        task.run()


def test_experiment_name(artifacts_server):
    with pytest.raises(ValueError):
        class InvalidTask(MlflowTask):
            config = TaskConfig(
                experiment_name="hi",  # noqa
                protocols=[DummyProtocol],
            )
    class ValidTask(MlflowTask):
        config = TaskConfig(
            protocols=[DummyProtocol],
        )

    assert ValidTask.get_experiment_name() == "ValidTask"
