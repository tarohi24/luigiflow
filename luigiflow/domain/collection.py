from dataclasses import dataclass, field
from typing import Optional, Union, cast

from luigiflow.domain.serializer import deserialize_params
from luigiflow.domain.tag_param import TaskParameter
from luigiflow.domain.task import MlflowTask
from luigiflow.task.protocol import MlflowTaskProtocol
from luigiflow.task.task_types import TaskImplementationList, TaskList


class TaskWithTheSameNameAlreadyRegistered(Exception):
    ...


class InconsistentDependencies(Exception):
    ...


class ProtocolNotRegistered(Exception):
    ...



@dataclass
class ProtocolCollectionItem:
    protocol_type: type[MlflowTaskProtocol]
    _task_class_dict: dict[str, type[MlflowTask]] = field(
        init=False, default_factory=dict
    )

    def register(self, task_class: type[MlflowTask]):
        key = task_class.__name__
        if key in self._task_class_dict:
            raise TaskWithTheSameNameAlreadyRegistered(
                f"{key} already registered in {self.protocol_type}"
            )
        self._task_class_dict[key] = task_class

    def get(self, task_name: str) -> type[MlflowTask]:
        return self._task_class_dict[task_name]


@dataclass(init=False)
class TaskCollectionImpl:
    """
    Note that this repository doesn't manage experiment names becuase that's not necessary.
    """

    _protocols: dict[str, ProtocolCollectionItem]

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
                    self._protocols[key] = ProtocolCollectionItem(prt)
                    self._protocols[key].register(task_cls)

    def generate_task_tree(
        self,
        task_params: TaskParameter,
        protocol: Optional[Union[str, type[MlflowTaskProtocol]]] = None,
    ) -> MlflowTask:
        if protocol is None:
            task_name: str = task_params["cls"]
            for repo_item in self._protocols.values():
                try:
                    repo_item.get(task_name)
                except KeyError:
                    continue
                # if successfully find a protocol
                protocol_name = repo_item.protocol_type.__name__
                break
            else:
                raise ValueError(f"Unknown task {task_name}")
        else:
            protocol_name = protocol if isinstance(protocol, str) else protocol.__name__
        cls_name = task_params["cls"]
        try:
            protocol_item = self._protocols[protocol_name]
        except KeyError:
            raise ProtocolNotRegistered(f"Unknown protocol: {protocol_name}")
        task_cls: type[MlflowTask] = protocol_item.get(cls_name)
        task_kwargs = deserialize_params(
            params=task_params.get("params", dict()),  # allow empty params
            param_types=task_cls.tag_manager.params,
        )
        # resolve requirements
        requirements: dict[
            str, type[MlflowTaskProtocol] | TaskList
        ] = task_cls.requirements
        requirements_required: dict[str, bool] = task_cls.requirements_required
        requirements_impl: dict[
            str, Union[MlflowTask, TaskImplementationList, None]
        ] = dict()
        if len(requirements) == 0:
            if (req := task_params.get("requires", None)) is not None:
                assert len(req) == 0
        else:
            assert "requires" in task_params
            # resolve its dependency
            for key, task_type in requirements.items():
                maybe_task_param = task_params["requires"][key]
                if maybe_task_param is None:
                    assert not requirements_required[key], f"{key} is required"
                    requirements_impl[key] = None
                elif isinstance(maybe_task_param, list):
                    assert isinstance(task_type, TaskList)
                    impls: list[MlflowTask] = []
                    for param in cast(list[TaskParameter], maybe_task_param):
                        req = self.generate_task_tree(
                            param,
                            protocol=task_type.protocol.__name__,
                        )
                        impls.append(req)
                    requirements_impl[key] = TaskImplementationList(impls)
                else:
                    assert isinstance(maybe_task_param, dict)
                    assert isinstance(task_type, type)
                    req_task_cls: MlflowTask = self.generate_task_tree(
                        cast(TaskParameter, maybe_task_param),
                        protocol=task_type.__name__,
                    )
                    requirements_impl[key] = req_task_cls
        return task_cls(requirements_impl=requirements_impl, **task_kwargs)
