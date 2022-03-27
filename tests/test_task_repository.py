from typing import Protocol, NoReturn, runtime_checkable

import pytest
from dependency_injector.wiring import inject, Provide

from luigiflow.task import MlflowTask, TaskConfig, MlflowTaskProtocol
from luigiflow.task_repository import TaskRepository, TaskWithTheSameNameAlreadyRegistered, InconsistentDependencies


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
        protocols=[DoNothingProtocol, ],
    )

    def _run(self) -> NoReturn:
        ...

    def do_nothing(self):
        ...


class NewTask(MlflowTask):
    config = TaskConfig(
        experiment_name="x",
        protocols=[AnotherProtocol, ],
    )

    @inject
    def requires(
        self,
        task_cls: type[MlflowTask] = Provide["DoNothingProtocol"],
    ) -> dict[str, MlflowTaskProtocol]:
        return {
            "1": task_cls(),
        }

    def _run(self) -> NoReturn:
        ...

    def method_a(self):
        ...


def test_duplicated_tasks():
    with pytest.raises(TaskWithTheSameNameAlreadyRegistered):
        TaskRepository(
            task_classes=[DoNothingImpl, DoNothingImpl, ],
            dependencies={
                "DoNothingProtocol": "DoNothingImpl",
            }
        )


def test_inconsistent_tasks_and_dependencies():
    with pytest.raises(InconsistentDependencies):
        TaskRepository(
            task_classes=[DoNothingImpl, ],
            dependencies={
                "dummy": "DoNothingImpl",  # invalid protocol name
            }
        )
    with pytest.raises(InconsistentDependencies):
        TaskRepository(
            task_classes=[DoNothingImpl, ],
            dependencies={
                "DoNothingProtocol": "dummy",  # invalid task name
            }
        )


def test_inject_dependencies():
    repo = TaskRepository(
        task_classes=[NewTask, DoNothingImpl],
        dependencies={
            "AnotherProtocol": "NewTask",
            "DoNothingProtocol": "DoNothingImpl",
        }
    )
    repo.inject_dependencies(module_to_wire=[__name__, ])


def test_ignore_missing_dependencies():
    TaskRepository(
        task_classes=[DoNothingImpl, NewTask],
        dependencies={
            # The dependency of AnotherTask is missing
            "DoNothingProtocol": "DoNothingImpl",
        },
        ignore_missing_dependencies=True,
    )
    with pytest.raises(InconsistentDependencies):
        TaskRepository(
            task_classes=[DoNothingImpl, NewTask],
            dependencies={
                # The dependency of AnotherTask is missing
                "DoNothingProtocol": "DoNothingImpl",
            },
            ignore_missing_dependencies=False,
        )
