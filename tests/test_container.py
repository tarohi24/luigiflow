from abc import ABC
from typing import NoReturn, Dict

import luigi
from dependency_injector.wiring import inject, Provide

from luigiflow.container import DiContainer
from luigiflow.interface import TaskInterface
from luigiflow.task import MlflowTask


class AbsClass(TaskInterface, ABC):

    @classmethod
    def get_experiment_name(cls) -> str:
        return "abs"

    @classmethod
    def get_artifact_filenames(cls) -> Dict[str, str]:
        return dict()

    def requires(self) -> Dict[str, luigi.Task]:
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
    def get_artifact_filenames(cls) -> Dict[str, str]:
        return dict()

    @inject
    def requires(
        self,
        abs_task: AbsClass = Provide["abs"],
    ) -> Dict[str, luigi.Task]:
        return {
            "dep": abs_task,
        }

    def _run(self) -> NoReturn:
        pass


def test_injection():
    container = DiContainer()
    container.load_dependencies(
        {
            "abs": "B",
        }
    )
    container.register_interface(AbsClass)
    container.activate_injection(modules=[__name__])
    dep_task = MainTask().requires()["dep"]
    assert isinstance(dep_task, ImplB)
