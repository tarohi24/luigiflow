from typing import Optional, Protocol

from luigiflow.domain.tag_param import TaskParameter
from luigiflow.domain.task import MlflowTask
from luigiflow.task.protocol import MlflowTaskProtocol


class TaskWithTheSameNameAlreadyRegistered(Exception):
    ...


class InconsistentDependencies(Exception):
    ...


class ProtocolNotRegistered(Exception):
    ...


class TaskCollection(Protocol):
    def generate_task_tree(
        self,
        task_params: TaskParameter,
        protocol: Optional[str | type[MlflowTaskProtocol]] = None,
    ) -> MlflowTask:
        raise NotImplementedError
