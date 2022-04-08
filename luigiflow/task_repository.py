from dataclasses import dataclass, field
from typing import Protocol, Any

from luigiflow.serializer import DESERIALIZERS
from luigiflow.task import MlflowTask
from luigiflow.types import TaskParameter


class TaskWithTheSameNameAlreadyRegistered(Exception):
    ...


class InconsistentDependencies(Exception):
    ...


def _deserialize_params(params: dict[str, Any], task_cls: type[MlflowTask]) -> dict[str, Any]:
    param_types = task_cls.param_types
    return {key: DESERIALIZERS[param_types[key].__name__](val) for key, val in params.items()}


@dataclass
class ProtocolRepositoryItem:
    protocol_type: type[Protocol]
    _task_class_dict: dict[str, type[MlflowTask]] = field(init=False, default_factory=dict)

    def register(self, task_class: type[MlflowTask]):
        if not issubclass(task_class, self.protocol_type):
            raise TypeError(f"{task_class} is not a {self.protocol_type}")
        key = task_class.__name__
        if key in self._task_class_dict:
            raise TaskWithTheSameNameAlreadyRegistered(f"{key} already registered in {self.protocol_type}")
        self._task_class_dict[key] = task_class

    def get(self, task_name: str) -> type[MlflowTask]:
        return self._task_class_dict[task_name]


@dataclass(init=False)
class TaskRepository:
    """
    Note that this repository doesn't manage experiment names becuase that's not necessary.
    """

    _protocols: dict[str, ProtocolRepositoryItem]

    def __init__(
        self,
        task_classes: list[type[MlflowTask]],
    ):
        # use for loop to check if there are duplicated names
        self._protocols = dict()
        for task_cls in task_classes:
            # register protocol
            for prt in task_cls.get_protocols():
                key = prt.__name__
                if key in self._protocols:
                    self._protocols[key].register(task_cls)
                else:
                    self._protocols[key] = ProtocolRepositoryItem(prt)
                    self._protocols[key].register(task_cls)

    def generate_task_tree(
        self,
        task_params: TaskParameter,
        protocol_name: str,
    ) -> MlflowTask:
        cls_name = task_params["cls"]
        task_cls = self._protocols[protocol_name].get(cls_name)
        task_kwargs = _deserialize_params(
            params=task_params["params"],
            task_cls=task_cls,
        )
        # resolve requirements
        if len(task_cls.requirements) == 0:
            assert "requires" not in task_params
        else:
            assert "requires" in task_params
            # resolve its dependency
            for key, protocol in task_cls.requirements.items():
                req_task_cls: MlflowTask = self.generate_task_tree(
                    task_params["requires"][key],
                    protocol_name=protocol.__name__,
                )
                task_cls.requirements_impl[key] = req_task_cls
        return task_cls(**task_kwargs)
