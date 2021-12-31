from abc import ABC
from typing import TypeVar, Type, Dict, cast

import luigi
from registrable import Registrable

from luigiflow.serializer import MlflowTagValue
from luigiflow.task import MlflowTask

T = TypeVar("T")


class TaskInterface(MlflowTask, Registrable, ABC):
    """
    The base class of each experiment (= a group of the tasks with the same output schema)
    should inherit this class.
    """

    @classmethod
    def get_subtask_name(cls) -> str:
        """
        Get name of this task within the interface.
        """
        raise NotImplementedError()

    def to_mlflow_tags(self) -> Dict[str, MlflowTagValue]:
        base = MlflowTask.to_mlflow_tags(self)
        base['name'] = self.get_subtask_name()
        return base


def resolve(
    cls: Type[MlflowTask],
    dependency_container_cls: Type[luigi.Config],
) -> Type[MlflowTask]:
    if issubclass(cls, TaskInterface):
        task_cls = cast(Type[TaskInterface], cls)
        dep = dependency_container_cls()
        task_name = task_cls.get_experiment_name()
        subtask_name = getattr(dep, task_name)
        if subtask_name is None:
            raise ValueError(f'Dependency of {task_name} is not specified')
        return task_cls.by_name(name=subtask_name)  # type: ignore
    else:
        # No need to resolve
        return cls
