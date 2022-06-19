import datetime
import json
import pickle
from typing import NoReturn, Optional, cast
from unittest import TestCase

import luigi
import pandas as pd
import pytest
from luigi import LocalTarget

from luigiflow.config import RunnerConfig
from luigiflow.runner import Runner
from luigiflow.task import (
    MlflowTask,
    TaskConfig,
    TryingToSaveUndefinedArtifact,
    MlflowTaskProtocol,
    TaskList,
    OptionalTask,
)
from luigiflow.task_repository import TaskRepository
from luigiflow.types import TaskParameter
from luigiflow.utils.savers import save_dataframe, save_pickle, save_json
from luigiflow.utils.testing import assert_two_tags_equal_wo_hashes


class DummyProtocol(MlflowTaskProtocol):
    ...


def test_to_mlflow_tags(monkeypatch):
    class Task(MlflowTask):
        param_int: int = luigi.IntParameter(default=10)
        param_str: str = luigi.Parameter(default="hi")
        param_bool: str = luigi.BoolParameter(default=True)
        param_date: datetime.date = luigi.DateParameter(default=datetime.date(2021, 1, 2))
        param_large_value: float = luigi.FloatParameter(default=2e11)
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
    actual = task.to_mlflow_tags()
    expected = {
        "name": "Task",
        "param_str": "hi",
        "param_bool": 1,
    }
    assert actual == expected

    # disable `tags_to_exclude`
    monkeypatch.setattr(Task, "tags_to_exclude", set())
    actual = task.to_mlflow_tags()
    expected = {
        "name": "Task",
        "param_int": 10,
        "param_str": "hi",
        "param_bool": 1,
        "param_date": "2021-01-02",
        "param_large_value": 200000000000.0,
    }
    assert actual == expected

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
    actual = main_task.to_mlflow_tags_w_parent_tags()
    expected = {
        "name": "MainTask",
        "_hash": "...",
        "bool_param": 0,
        "ccc.name": "TaskC",
        "ccc._hash": "...",
        "ccc.int_param": 10,
        "bbb.name": "TaskB",
        "bbb._hash": "...",
        "bbb.value": 1,
        "bbb.aaa.name": "TaskA",
        "bbb.aaa._hash": "...",
        "bbb.aaa.param": "hi",
    }
    assert_two_tags_equal_wo_hashes(actual, expected)


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


def test_to_mlflow_tags_with_non_mlflow_task_requirements(tmpdir, artifacts_server):
    class TaskA(MlflowTask):
        value: int = luigi.IntParameter()
        config = TaskConfig(
            protocols=[DummyProtocol],
        )

        def _run(self) -> NoReturn:
            self.save_to_mlflow()

    class TaskB(MlflowTask):
        config = TaskConfig(
            protocols=[DummyProtocol],
            requirements={
                "a": TaskList(DummyProtocol),
                "b": OptionalTask(DummyProtocol),
            },
        )

        def _run(self) -> NoReturn:
            self.save_to_mlflow()

    config = {
        "cls": "TaskB",
        "requires": {
            "a": [
                {
                    "cls": "TaskA",
                    "params": {
                        "value": 1,
                    },
                },
                {
                    "cls": "TaskA",
                    "params": {
                        "value": 2,
                    },
                },
            ],
            "b": None,
        },
    }
    d = tmpdir.join("d")
    d.mkdir()
    config_path = d.join("config.json")
    with open(config_path, "w") as fout:
        json.dump(config, fout)
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
    task, _ = runner.run(
        protocol_name="DummyProtocol",
        config_jsonnet_path=config_path,
        dry_run=True,
    )
    actual_tags = cast(MlflowTask, task).to_mlflow_tags_w_parent_tags()
    expected_tags = {
        "name": "TaskB",
        "_hash": "...",
        "a.0.name": "TaskA",
        "a.0.value": 1,
        "a.0._hash": "...",
        "a.1.name": "TaskA",
        "a.1.value": 2,
        "a.1._hash": "...",
        # "b" should not appear
    }
    assert_two_tags_equal_wo_hashes(actual_tags, expected_tags)


def test_too_many_mlflow_tags(artifacts_server):
    class TaskA(MlflowTask):
        value: int = luigi.IntParameter()
        config = TaskConfig(
            protocols=[DummyProtocol],
        )

        def _run(self) -> NoReturn:
            self.save_to_mlflow()

    class TaskB(MlflowTask):
        value: int = luigi.IntParameter()
        config = TaskConfig(
            protocols=[DummyProtocol],
            requirements={
                "a": TaskList(DummyProtocol),
            },
        )

        def _run(self) -> NoReturn:
            self.save_to_mlflow()

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
    task_param: TaskParameter = {
        "cls": "TaskB",
        "params": {
            "value": 1,
        },
        "requires": {
            "a": [
                {
                    "cls": "TaskA",
                    "params": {
                        "value": i,
                    },
                }
                for i in range(200)
            ]
        },
    }
    task, _ = runner.run_with_task_param(
        protocol_name="DummyProtocol",
        task_param=task_param,
        dry_run=True,
    )
    actual = cast(MlflowTask, task).to_mlflow_tags_w_parent_tags()
    expected = {
        "name": "TaskB",
        "value": 1,
        "a_hash": "cd16cc266124a7893d2b257ab61fba78",
        "_hash": "dd3815e2b87846ed47d0614cf2daba64",
    }
    assert expected == actual

    # hash values should not change each time
    actual = cast(MlflowTask, task).to_mlflow_tags_w_parent_tags()
    assert expected == actual

    task_param["requires"]["a"][0]["params"]["value"] = 1  # modify
    task, _ = runner.run_with_task_param(
        protocol_name="DummyProtocol",
        task_param=task_param,
        dry_run=True,
    )
    actual = cast(MlflowTask, task).to_mlflow_tags_w_parent_tags()
    assert expected["a_hash"] != actual["a_hash"]

    task_param: TaskParameter = {
        "cls": "TaskB",
        "params": {
            "value": 1,
        },
        "requires": {
            "a": [
                {
                    "cls": "TaskA",
                    "params": {
                        "value": i,
                    },
                }
                for i in range(201)  # changed
            ]
        },
    }
    task, _ = runner.run_with_task_param(
        protocol_name="DummyProtocol",
        task_param=task_param,
        dry_run=True,
    )
    actual = cast(MlflowTask, task).to_mlflow_tags_w_parent_tags()
    assert set(actual.keys()) == set(expected.keys())
    assert expected["a_hash"] != actual["a_hash"]


def test_hash_of_nested_requirements(artifacts_server):
    class TaskA(MlflowTask):
        value: int = luigi.IntParameter()
        config = TaskConfig(
            protocols=[DummyProtocol],
        )

        def _run(self) -> NoReturn:
            self.save_to_mlflow()

    class TaskB(MlflowTask):
        value: int = luigi.IntParameter()
        config = TaskConfig(
            protocols=[DummyProtocol],
            requirements={
                "a": TaskList(DummyProtocol),
            },
        )

        def _run(self) -> NoReturn:
            self.save_to_mlflow()

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
    task_param: TaskParameter = {
        "cls": "TaskB",
        "params": {
            "value": 1,
        },
        "requires": {
            "a": [
                {
                    "cls": "TaskB",
                    "params": {
                        "value": 1,
                    },
                    "requires": {
                        "a": [
                            {
                                "cls": "TaskA",
                                "params": {
                                    "value": 1,
                                },
                            }
                        ]
                    }
                }
            ]
        },
    }
    task, _ = runner.run_with_task_param(
        protocol_name="DummyProtocol",
        task_param=task_param,
        dry_run=True,
    )
    hash_before = cast(MlflowTask, task).to_mlflow_tags_w_parent_tags()["a.0._hash"]
    task_param["requires"]["a"][0]["requires"]["a"][0]["params"]["value"] = 2
    task, _ = runner.run_with_task_param(
        protocol_name="DummyProtocol",
        task_param=task_param,
        dry_run=True,
    )
    hash_after = cast(MlflowTask, task).to_mlflow_tags_w_parent_tags()["a.0._hash"]
    assert hash_before != hash_after

