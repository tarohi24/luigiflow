from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Type

from dependency_injector import providers
from dependency_injector.containers import DynamicContainer

from luigiflow.config.jsonnet import JsonnetConfigLoader
from luigiflow.task import MlflowTask


class TaskWithTheSameNameAlreadyRegistered(Exception):
    ...


class InconsistentDependencies(Exception):
    ...


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
    dependencies: dict[str, str]

    def __init__(
        self,
        task_classes: list[type[MlflowTask]],
        dependencies: dict[str, str],
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

        self.dependencies = dependencies
        dep_keys = set(self.dependencies.keys())
        prt_keys = set(self._protocols.keys())
        if len(diff := (dep_keys - prt_keys)) > 0:
            raise InconsistentDependencies(f"Unknown dependencies: {diff}")
        if len(diff := (prt_keys - dep_keys)) > 0:
            raise InconsistentDependencies(f"Dependencies not specified: {diff}")
        for protocol_name, task_name in self.dependencies.items():
            try:
                self._protocols[protocol_name].get(task_name)
            except KeyError:
                raise InconsistentDependencies(f"{task_name} not registered to {protocol_name}")

    def inject_dependencies(self, module_to_wire: list[str] = None):
        module_to_wire = module_to_wire or []
        # inject dependencies
        container = DynamicContainer()
        for protocol_name, repo in self._protocols.items():
            default_task: type[MlflowTask] = repo.get(self.dependencies[protocol_name])
            # Don't use `providers.Factory(lambda: default)` since it returns the latest `default`,
            # not the one at the time that `default` is specified
            setattr(container, protocol_name, providers.Object(default_task))
        container.wire(module_to_wire)

    def generate_tasks(
        self,
        protocol_name: str,
        params: list[dict[str, Any]],
        context_config_path: Path,
    ) -> list[MlflowTask]:
        if not isinstance(params, list):
            raise ValueError("Pass a list of kwargs `params`")
        proto = self._protocols[protocol_name]
        task_class = proto.get(self.dependencies[protocol_name])
        tasks = []
        if len(params) > 0:
            for param in params:
                config_loader = JsonnetConfigLoader(external_variables=param)
                with config_loader.load(context_config_path):
                    task = task_class()
                    tasks.append(task)
        else:
            # no external params
            config_loader = JsonnetConfigLoader()
            with config_loader.load(context_config_path):
                task = task_class()
                tasks.append(task)
        return tasks
