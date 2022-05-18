import json
from pathlib import Path
from typing import Protocol, NoReturn, runtime_checkable

import luigi
import pytest

from luigiflow.config import RunnerConfig
from luigiflow.runner import Runner
from luigiflow.task import MlflowTask, TaskConfig, MlflowTaskProtocol
from luigiflow.task_repository import (
    TaskRepository,
    TaskWithTheSameNameAlreadyRegistered,
    ProtocolNotRegistered,
)
from luigiflow.utils.savers import save_json


@runtime_checkable
class AProtocol(MlflowTaskProtocol, Protocol):
    def do_nothing(self):
        raise NotImplementedError()


@runtime_checkable
class AnotherProtocol(MlflowTaskProtocol, Protocol):
    def method_a(self):
        raise NotImplementedError()


class DoNothingImpl(MlflowTask):
    config = TaskConfig(
        experiment_name="do_nothing",
        protocols=[
            AProtocol,
        ],
        requirements=dict(),
    )

    def _run(self) -> NoReturn:
        ...

    def do_nothing(self):
        ...


class NewTask(MlflowTask):
    param: str = luigi.Parameter()
    config = TaskConfig(
        experiment_name="x",
        protocols=[
            AnotherProtocol,
        ],
        requirements={
            "1": AProtocol,
        },
    )

    def _run(self) -> NoReturn:
        self.save_to_mlflow()

    def method_a(self):
        ...


def test_duplicated_tasks():
    with pytest.raises(TaskWithTheSameNameAlreadyRegistered):
        TaskRepository(
            task_classes=[
                DoNothingImpl,
                DoNothingImpl,
            ],
        )


def test_unknown_protocol():
    class UnknownProtocol(MlflowTaskProtocol):
        ...

    class UnknownTask(MlflowTask):
        config = TaskConfig(
            experiment_name="hi",
            protocols=[
                UnknownProtocol,
            ],
            requirements=dict(),
        )

    class TaskHavingUnknownProtocol(MlflowTask):
        config = TaskConfig(
            experiment_name="hi",
            protocols=[
                AProtocol,
            ],
            requirements={
                "unknown": UnknownProtocol,
            },
        )

    repo = TaskRepository(
        task_classes=[
            TaskHavingUnknownProtocol,
        ],
    )
    with pytest.raises(ProtocolNotRegistered):
        repo.generate_task_tree(
            task_params={
                "cls": "TaskHavingUnknownProtocol",
                "params": dict(),
                "requires": {
                    "unknown": {
                        "cls": "UnknownTask",
                        "params": dict(),
                    },
                },
            },
            protocol="DoNothingProtocol",
        )


def test_recursively_nested_task(artifacts_server):
    class TaskA(MlflowTask):
        param: str = luigi.Parameter()
        config = TaskConfig(
            experiment_name="hi",
            protocols=[
                AProtocol,
            ],
            requirements={
                "req": AProtocol,
            },
            artifact_filenames=dict(),
        )

    class TaskB(MlflowTask):
        config = TaskConfig(
            experiment_name="hi",
            protocols=[
                AProtocol,
            ],
            requirements={
                "req": AProtocol,
            },
            artifact_filenames=dict(),
        )

    class TaskC(MlflowTask):
        config = TaskConfig(
            experiment_name="hi",
            protocols=[
                AProtocol,
            ],
            requirements=dict(),
            artifact_filenames=dict(),
        )

    repo = TaskRepository(
        task_classes=[
            TaskA,
            TaskB,
            TaskC,
        ],
    )
    task: MlflowTask = repo.generate_task_tree(
        task_params={
            "cls": "TaskA",
            "params": {
                "param": "top",
            },
            "requires": {
                "req": {
                    "cls": "TaskB",
                    "requires": {
                        "req": {
                            "cls": "TaskA",
                            "params": {
                                "param": "bottom",
                            },
                            "requires": {
                                "req": {
                                    "cls": "TaskB",
                                    "requires": {
                                        "req": {
                                            "cls": "TaskC",
                                        }
                                    }
                                },
                            },
                        }
                    }
                }
            },
        },
        protocol="AProtocol",
    )
    tags = task.to_mlflow_tags_w_parent_tags()
    assert tags["param"] == "top"


def test_too_many_tags(artifacts_server, tmpdir):

    class DoSomething(MlflowTask):
        value: str = luigi.IntParameter(default=1)
        config = TaskConfig(
            experiment_name="hi",
            protocols=[
                AProtocol,
            ],
            requirements=dict(),
            artifact_filenames=dict(
                out="out.jsonl",
            ),
        )

        def _run(self) -> NoReturn:
            self.save_to_mlflow(
                artifacts_and_save_funcs={
                    "out": (dict(), save_json),
                }
            )

    runner = Runner(
        config=RunnerConfig(
            mlflow_tracking_uri=artifacts_server.url,
            use_local_scheduler=True,
            create_experiment_if_not_existing=True,
        ),
        experiment_repository=TaskRepository(
            task_classes=[DoSomething, NewTask],
        ),
    )
    d = Path(tmpdir.join("sub"))
    d.mkdir()
    config_path = d / "config.jsonnet"
    with open(config_path, "w") as fout:
        json.dump(
            {
                "cls": "NewTask",
                "params": {
                    "param": "test" * 500,  # a long query
                },
                "requires": {
                    "1": {
                        "cls": "DoSomething",
                    }
                }
            },
            fout
        )
    task, res = runner.run(
        protocol_name="AnotherProtocol",
        config_jsonnet_path=config_path,
        dry_run=False,
    )
    assert task.complete()
