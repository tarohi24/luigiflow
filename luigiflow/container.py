from dataclasses import dataclass, field
from typing import Type, List, Dict

from dependency_injector import providers
from dependency_injector.containers import DynamicContainer

from luigiflow.config import ConfigContext
from luigiflow.interface import TaskInterface
from luigiflow.task import MlflowTask


class InvalidOperation(Exception):
    pass


def initialize_container() -> DynamicContainer():
    container = DynamicContainer()
    container.config = providers.Configuration()
    return container


def resolve_task(interface_cls: Type[TaskInterface], subtask_name: str) -> Type[MlflowTask]:
    if subtask_name is None:
        raise ValueError(f"Dependency of {interface_cls.get_experiment_name()} is null")
    return interface_cls.by_name(subtask_name)


@dataclass
class DiContainer:
    container: DynamicContainer = field(default_factory=initialize_container)
    is_activated: bool = field(init=False)

    def __post_init__(self):
        self.is_activated = False

    def register_interface(self, interface_cls: Type[TaskInterface]) -> 'DiContainer':
        if self.is_activated:
            raise InvalidOperation("You cannot register a new interface after activating the container")
        setattr(
            self.container,
            interface_cls.get_experiment_name(),
            providers.Callable(
                resolve_task,
                interface_cls=interface_cls,
                subtask_name=getattr(self.container.config.dependencies, interface_cls.get_experiment_name()),
            )
        )
        return self

    def activate_injection(self, modules: List[str]) -> 'DiContainer':
        self.is_activated = True
        self.container.wire(modules=modules)
        return self

    def load_dependencies(self, dependencies: Dict[str, str]) -> 'DiContainer':
        """
        :param dependencies: `{experiemnt_name: subtask_name}`
        :return: self
        """
        if self.is_activated:
            raise InvalidOperation("You cannot add dependencies after activating the container.")
        self.container.config.from_dict(
            {
                "dependencies": dependencies,
            }
        )
        return self
