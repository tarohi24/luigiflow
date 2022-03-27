from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from luigiflow.config.jsonnet import JsonnetConfigLoader
from luigiflow.task import MlflowTask


@dataclass
class ProtocolRepositoryItem:
    protocol_type: type[Protocol]
    _task_class_dict: dict[str, type[MlflowTask]] = field(init=False, default_factory=dict)

    def register(self, task_class: type[MlflowTask]):
        if not issubclass(task_class, self.protocol_type):
            raise ValueError(f"{task_class} is not a {self.protocol_type}")
        key = task_class.__name__
        if key in self._task_class_dict:
            raise ValueError(f"{key} already registered in {self.protocol_type}")
        self._task_class_dict[key] = task_class

    def get(self, task_name: str) -> type[MlflowTask]:
        return self._task_class_dict[task_name]


@dataclass(init=False)
class TaskRepository:
    _experiments_dict: dict[str, dict[str, type[MlflowTask]]]
    _protocols: dict[str, ProtocolRepositoryItem]
    dependencies: dict[str, str]

    def __init__(
        self,
        task_classes: list[type[MlflowTask]],
        dependencies: dict[str, str],
    ):
        # use for loop to check if there are duplicated names
        self._experiments_dict = defaultdict(dict)
        self._protocols = dict()
        for task_cls in task_classes:
            # register experiment
            exp_name = task_cls.get_experiment_name()
            sub_dict = self._experiments_dict[exp_name]
            sub_name = task_cls.get_subtask_name()
            if sub_name in sub_dict:
                raise ValueError(f"Duplicated (experiment, sub_experiment) = {(exp_name, sub_name)}")
            sub_dict[sub_name] = task_cls
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
            raise ValueError(f"Unknown dependencies: {diff}")
        if len(diff := (prt_keys - dep_keys)) > 0:
            raise ValueError(f"Dependencies not specified: {diff}")

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
