from abc import ABC
from typing import NoReturn, Dict

import luigi
import pandas as pd
from dependency_injector import providers
from dependency_injector.containers import DeclarativeContainer
from dependency_injector.wiring import inject, Provide

from luigiflow.config import JsonnetConfigLoader
from luigiflow.interface import TaskInterface
from luigiflow.savers import save_dataframe
from luigiflow.task import MlflowTask


class AbstractTask(TaskInterface, ABC):

    @classmethod
    def get_experiment_name(cls) -> str:
        return "dummy"

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
        abstract_task: AbstractTask = Provide["dummy_task"],
    ) -> Dict[str, luigi.Task]:
        return {
            "dummy": abstract_task,
        }

    def _run(self) -> NoReturn:
        pass


class Container(DeclarativeContainer):
    config = providers.Configuration()
    dummy_task = providers.Singleton(
        lambda subname: AbstractTask.by_name(subname)(),
        subname=config.dependencies.dummy,
    )


def test_change_implementation(tmpdir):
    d = tmpdir.mkdir("d")
    config_path = d.join("config.jsonnet")
    with config_path.open("w") as fout:
        fout.write("""
        {
            "dependencies": {
                "dummy": "B",
            }
        }
        """)
    config_loader = JsonnetConfigLoader()
    with config_loader.load(config_path) as context:
        container = Container()
        container.config.from_dict(
            {
                "dependencies": context.get_dependencies(),
            }
        )
        container.wire(modules=[__name__, ])
        task = AnotherTask()
        assert isinstance(task.requires()['dummy'], ImplB)
