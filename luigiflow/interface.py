from abc import ABC
from typing import TypeVar, cast

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

    def to_mlflow_tags(self) -> dict[str, MlflowTagValue]:
        base = MlflowTask.to_mlflow_tags(self)
        base['name'] = self.get_subtask_name()
        return base
