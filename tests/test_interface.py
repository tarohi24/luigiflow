from abc import ABC
from typing import NoReturn, Dict, Type

import luigi
import pandas as pd
from dependency_injector.wiring import inject, Provide

from luigiflow.config import JsonnetConfigLoader
from luigiflow.container import DiContainer
from luigiflow.interface import TaskInterface
from luigiflow.savers import save_dataframe
from luigiflow.task import MlflowTask


class AbstractTask(TaskInterface, ABC):

    @classmethod
    def get_experiment_name(cls) -> str:
        return "dummy_exp"

    @classmethod
    def get_artifact_filenames(cls) -> Dict[str, str]:
        return {
            "dummy": "dummy.csv",
        }

    def requires(self) -> Dict[str, luigi.Task]:
        return dict()


@AbstractTask.register("A")
class ImplA(AbstractTask):

    @classmethod
    def get_subtask_name(cls) -> str:
        return "a"

    def _run(self) -> NoReturn:
        df = pd.DataFrame([
            {"label": "A"},
        ])
        self.save_to_mlflow(
            artifacts_and_save_funcs={
                "dummy": (df, save_dataframe),
            }
        )


@AbstractTask.register("B")
class ImplB(AbstractTask):

    @classmethod
    def get_subtask_name(cls) -> str:
        return "b"

    def _run(self) -> NoReturn:
        df = pd.DataFrame([
            {"label": "B"},
        ])
        self.save_to_mlflow(
            artifacts_and_save_funcs={
                "dummy": (df, save_dataframe),
            }
        )


class AnotherTask(MlflowTask):

    @classmethod
    def get_experiment_name(cls) -> str:
        pass

    @classmethod
    def get_artifact_filenames(cls) -> Dict[str, str]:
        pass

    @inject
    def requires(
        self,
        abstract_task_cls: Type[AbstractTask] = Provide["dummy_exp"],
    ) -> Dict[str, luigi.Task]:
        return {
            "dummy": abstract_task_cls(),
        }

    def _run(self) -> NoReturn:
        pass


def test_change_implementation(tmpdir):
    d = tmpdir.mkdir("d")
    config_path = d.join("config.jsonnet")
    with config_path.open("w") as fout:
        fout.write("""
        {
            "dependencies": {
                "dummy_exp": "B",
            }
        }
        """)
    config_loader = JsonnetConfigLoader()
    with config_loader.load(config_path) as context:
        container = DiContainer()
        container.load_dependencies(dependencies=context.get_dependencies())
        container.register_interface(AbstractTask)
        container.activate_injection(modules=[__name__, ])
        task = AnotherTask()
        assert isinstance(task.requires()['dummy'], ImplB)
