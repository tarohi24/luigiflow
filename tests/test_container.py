from abc import ABC
from typing import NoReturn, cast

import luigi
import pytest
from dependency_injector.wiring import inject, Provide

from luigiflow.container import TaskDependencyResolver, InvalidOperation, CannotSolveDependency
from luigiflow.interface import TaskInterface
from luigiflow.task import MlflowTask


class AbsClass(TaskInterface, ABC):

    @classmethod
    def get_experiment_name(cls) -> str:
        return "abs"

    @classmethod
    def get_artifact_filenames(cls) -> dict[str, str]:
        return dict()

    def requires(self) -> dict[str, luigi.Task]:
        return dict()

    def _run(self) -> NoReturn:
        self.save_to_mlflow(
            metrics={
                "value": 0.1,
            }
        )


@AbsClass.register("A")
class ImplA(AbsClass):

    @classmethod
    def get_subtask_name(cls) -> str:
        return "A"


@AbsClass.register("B")
class ImplB(AbsClass):

    @classmethod
    def get_subtask_name(cls) -> str:
        return "B"


class MainTask(MlflowTask):

    @classmethod
    def get_experiment_name(cls) -> str:
        return "main"

    @classmethod
    def get_artifact_filenames(cls) -> dict[str, str]:
        return dict()

    @inject
    def requires(
        self,
        abs_task_cls: type[AbsClass] = Provide["abs"],
    ) -> dict[str, luigi.Task]:
        return {
            "dep": abs_task_cls(),
        }

    def _run(self) -> NoReturn:
        pass


def test_injection():
    container = TaskDependencyResolver()
    container.load_dependencies(
        {
            "abs": "B",
        }
    )
    container.register_interface(AbsClass)
    container.activate_injection(modules=[__name__])
    dep_task = MainTask().requires()["dep"]
    assert cast(ImplB, dep_task).get_subtask_name() == "B"  # Don't test with `isinstance`

    # test with modifying order of the initialization
    new_container = TaskDependencyResolver()
    new_container.load_dependencies(
        {
            "abs": "A",
        }
    )
    new_container.register_interface(AbsClass)
    new_container.activate_injection(modules=[__name__])
    dep_task = MainTask().requires()["dep"]
    assert cast(ImplB, dep_task).get_subtask_name() == "A"  # Don't test with `isinstance`
    assert isinstance(dep_task, TaskInterface)


def test_invalid_injection_order():
    container = TaskDependencyResolver()
    container.activate_injection(modules=[__name__])
    with pytest.raises(InvalidOperation):
        container.register_interface(AbsClass)
    with pytest.raises(InvalidOperation):
        container.load_dependencies(
            {
                "abs": "B",
            }
        )


def test_resolve_interface():
    container = TaskDependencyResolver()
    with pytest.raises(CannotSolveDependency):
        container.resolve(AbsClass)
    container.register_interface(AbsClass)
    with pytest.raises(CannotSolveDependency):
        container.resolve(AbsClass)
    container.load_dependencies({"abs": "B"})
    sub_class = container.resolve(AbsClass)
    assert issubclass(container.resolve(AbsClass), ImplB)
    container.load_dependencies({"abs": "A"})
    assert issubclass(container.resolve(AbsClass), ImplA)
