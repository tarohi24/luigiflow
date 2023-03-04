from typing import Optional, Protocol

from luigiflow.domain.tag_manager import TaskParameter
from luigiflow.domain.task import DeprecatedTaskProtocol
from luigiflow.infrastructure.luigi.task import DeprecatedTask


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
        protocol: Optional[str | type[DeprecatedTaskProtocol]] = None,
    ) -> DeprecatedTask:
        raise NotImplementedError
