import json
import pickle
from typing import NoReturn, Optional, cast

import luigi
import pandas as pd
import pytest
from luigi import LocalTarget

from luigiflow.config import RunnerConfig
from luigiflow.domain.tag_manager import TaskParameter
from luigiflow.domain.task import (
    DeprecatedTaskProtocol,
    OptionalTask,
    TaskConfig,
    TaskList,
    TryingToSaveUndefinedArtifact,
)
from luigiflow.infrastructure.luigi.task import DeprecatedTask
from luigiflow.infrastructure.mlflow.collection import TaskCollectionImpl
from luigiflow.infrastructure.mlflow.task_run import MlflowTaskRunRepository
from luigiflow.utils.savers import save_dataframe, save_json, save_pickle
from luigiflow.utils.testing import assert_two_tags_equal_wo_hashes


class DummyProtocolDeprecated(DeprecatedTaskProtocol):
    ...


def test_to_tags_w_parents(monkeypatch):
    class IDeprecatedTaskA(DeprecatedTaskProtocol):
        ...

    class IDeprecatedTaskB(DeprecatedTaskProtocol):
        ...

    class IDeprecatedTaskC(DeprecatedTaskProtocol):
        ...

    class IMainDeprecatedTask(DeprecatedTaskProtocol):
        ...

    class TaskA(DeprecatedTask[dict]):
        param: str = luigi.Parameter(default="hi")
        config = TaskConfig(
            protocols=[
                IDeprecatedTaskA,
            ],
            requirements=dict(),
        )

        def _run(self) -> NoReturn:
            ...

    class TaskB(DeprecatedTask[dict]):
        value: int = luigi.IntParameter(default=1)
        config = TaskConfig(
            protocols=[
                IDeprecatedTaskB,
            ],
            requirements={
                "aaa": IDeprecatedTaskA,
            },
        )

        def _run(self) -> NoReturn:
            ...

    class TaskC(DeprecatedTask[dict]):
        int_param: int = luigi.IntParameter(default=10)
        config = TaskConfig(
            protocols=[
                IDeprecatedTaskC,
            ],
            requirements=dict(),
        )

        def _run(self) -> NoReturn:
            ...

    class MainTask(DeprecatedTask[dict]):
        bool_param: bool = luigi.BoolParameter(default=False)
        config = TaskConfig(
            protocols=[
                IMainDeprecatedTask,
            ],
            requirements={
                "bbb": IDeprecatedTaskB,
                "ccc": IDeprecatedTaskC,
            },
        )

        def _run(self) -> NoReturn:
            ...

    task_repo = TaskCollectionImpl(
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

    class Task(Task):
        config = TaskConfig(
            artifact_filenames={
                "csv": "df.csv",
                "pickle": "df.pickle",
                "text": "text.txt",
            },
            protocols=[
                DummyProtocolDeprecated,
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

    task = TaskCollectionImpl([Task,]).generate_task_tree(
        task_params={
            "cls": "Task",
        },
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
    class InvalidTask(DeprecatedTask):
        config = TaskConfig(
            protocols=[
                DummyProtocolDeprecated,
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

    task = TaskCollectionImpl([InvalidTask,]).generate_task_tree(
        task_params={"cls": "InvalidTask"},
        protocol=DummyProtocolDeprecated,
    )
    assert task.output() is None
    with pytest.raises(TryingToSaveUndefinedArtifact):
        task.run()


def test_experiment_name(artifacts_server):
    with pytest.raises(ValueError):

        class InvalidTask(DeprecatedTask):
            config = TaskConfig(
                experiment_name="hi",  # noqa
                protocols=[DummyProtocolDeprecated],
            )

    class ValidTask(DeprecatedTask):
        config = TaskConfig(
            protocols=[DummyProtocolDeprecated],
        )

    assert ValidTask.get_experiment_name() == "ValidTask"


def test_to_mlflow_tags_with_non_mlflow_task_requirements(tmpdir, artifacts_server):
    class TaskA(DeprecatedTask):
        value: int = luigi.IntParameter()
        config = TaskConfig(
            protocols=[DummyProtocolDeprecated],
        )

        def _run(self) -> NoReturn:
            self.save_to_mlflow()

    class TaskB(DeprecatedTask):
        config = TaskConfig(
            protocols=[DummyProtocolDeprecated],
            requirements={
                "a": TaskList(DummyProtocolDeprecated),
                "b": OptionalTask(DummyProtocolDeprecated),
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
    runner = MlflowTaskRunRepository(
        config=RunnerConfig(
            mlflow_tracking_uri=artifacts_server.url,
            use_local_scheduler=True,
            create_experiment_if_not_existing=True,
        ),
        experiment_repository=TaskCollectionImpl(
            task_classes=[TaskA, TaskB],
        ),
    )
    task, _ = runner.run(
        config_jsonnet_path=config_path,
        dry_run=True,
    )
    actual_tags = cast(DeprecatedTask, task).to_mlflow_tags_w_parent_tags()
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
    class TaskA(DeprecatedTask):
        value: int = luigi.IntParameter()
        config = TaskConfig(
            protocols=[DummyProtocolDeprecated],
        )

        def _run(self) -> NoReturn:
            self.save_to_mlflow()

    class TaskB(DeprecatedTask):
        value: int = luigi.IntParameter()
        config = TaskConfig(
            protocols=[DummyProtocolDeprecated],
            requirements={
                "a": TaskList(DummyProtocolDeprecated),
            },
        )

        def _run(self) -> NoReturn:
            self.save_to_mlflow()

    runner = MlflowTaskRunRepository(
        config=RunnerConfig(
            mlflow_tracking_uri=artifacts_server.url,
            use_local_scheduler=True,
            create_experiment_if_not_existing=True,
        ),
        experiment_repository=TaskCollectionImpl(
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
        task_param=task_param,
        dry_run=True,
    )
    actual = cast(DeprecatedTask, task).to_mlflow_tags_w_parent_tags()
    expected = {
        "name": "TaskB",
        "value": 1,
        "a_hash": "cd16cc266124a7893d2b257ab61fba78",
        "_hash": "dd3815e2b87846ed47d0614cf2daba64",
    }
    assert expected == actual

    # hash values should not change each time
    actual = cast(DeprecatedTask, task).to_mlflow_tags_w_parent_tags()
    assert expected == actual

    task_param["requires"]["a"][0]["params"]["value"] = 1  # modify
    task, _ = runner.run_with_task_param(
        task_param=task_param,
        dry_run=True,
    )
    actual = cast(DeprecatedTask, task).to_mlflow_tags_w_parent_tags()
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
        task_param=task_param,
        dry_run=True,
    )
    actual = cast(DeprecatedTask, task).to_mlflow_tags_w_parent_tags()
    assert set(actual.keys()) == set(expected.keys())
    assert expected["a_hash"] != actual["a_hash"]


def test_hash_of_nested_requirements(artifacts_server):
    class TaskA(DeprecatedTask):
        value: int = luigi.IntParameter()
        config = TaskConfig(
            protocols=[DummyProtocolDeprecated],
        )

        def _run(self) -> NoReturn:
            self.save_to_mlflow()

    class TaskB(DeprecatedTask):
        value: int = luigi.IntParameter()
        config = TaskConfig(
            protocols=[DummyProtocolDeprecated],
            requirements={
                "a": TaskList(DummyProtocolDeprecated),
            },
        )

        def _run(self) -> NoReturn:
            self.save_to_mlflow()

    runner = MlflowTaskRunRepository(
        config=RunnerConfig(
            mlflow_tracking_uri=artifacts_server.url,
            use_local_scheduler=True,
            create_experiment_if_not_existing=True,
        ),
        experiment_repository=TaskCollectionImpl(
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
                    },
                }
            ]
        },
    }
    task, _ = runner.run_with_task_param(
        task_param=task_param,
        dry_run=True,
    )
    hash_before = cast(DeprecatedTask, task).to_mlflow_tags_w_parent_tags()["a.0._hash"]
    task_param["requires"]["a"][0]["requires"]["a"][0]["params"]["value"] = 2
    task, _ = runner.run_with_task_param(
        task_param=task_param,
        dry_run=True,
    )
    hash_after = cast(DeprecatedTask, task).to_mlflow_tags_w_parent_tags()["a.0._hash"]
    assert hash_before != hash_after
