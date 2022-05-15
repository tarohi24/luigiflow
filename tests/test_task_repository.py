from typing import Protocol, NoReturn, runtime_checkable

import luigi
import pytest

from luigiflow.task import MlflowTask, TaskConfig, MlflowTaskProtocol
from luigiflow.task_repository import (
    TaskRepository,
    TaskWithTheSameNameAlreadyRegistered,
    ProtocolNotRegistered,
)


@runtime_checkable
class DoNothingProtocol(MlflowTaskProtocol, Protocol):
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
            DoNothingProtocol,
        ],
        requirements=dict(),
    )

    def _run(self) -> NoReturn:
        ...

    def do_nothing(self):
        ...


class NewTask(MlflowTask):
    config = TaskConfig(
        experiment_name="x",
        protocols=[
            AnotherProtocol,
        ],
        requirements={
            "1": DoNothingProtocol,
        },
    )

    def _run(self) -> NoReturn:
        ...

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
                DoNothingProtocol,
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


def test_recursively_nested_task():
    class TaskA(MlflowTask):
        param: str = luigi.Parameter()
        config = TaskConfig(
            experiment_name="hi",
            protocols=[
                DoNothingProtocol,
            ],
            requirements={
                "req": DoNothingProtocol,
            },
            artifact_filenames=dict(),
        )

    class TaskB(MlflowTask):
        config = TaskConfig(
            experiment_name="hi",
            protocols=[
                DoNothingProtocol,
            ],
            requirements={
                "req": DoNothingProtocol,
            },
            artifact_filenames=dict(),
        )

    class TaskC(MlflowTask):
        config = TaskConfig(
            experiment_name="hi",
            protocols=[
                DoNothingProtocol,
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
    repo.generate_task_tree(
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
                                    "cls": "TaskC",
                                },
                            },
                        }
                    }
                }
            },
        },
        protocol="DoNothingProtocol",
    )
