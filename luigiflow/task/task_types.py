from dataclasses import dataclass
from typing import Protocol, Generic, Callable, Iterator, Union, Any, TypeVar

import luigi
from luigi.task_register import Register

from luigiflow.task.protocol import MlflowTaskProtocol

V = TypeVar("V", bound=type[MlflowTaskProtocol])
K = TypeVar("K")
T = TypeVar("T", bound=dict)  # to denote the type of `task.requires()`


@dataclass
class OptionalTask:
    base_cls: type[Protocol]

    def __post_init__(self):
        assert issubclass(self.base_cls, MlflowTaskProtocol)


@dataclass
class TaskList(Generic[V]):
    protocol: type[V]

    # this method is just to give hints
    def apply(self, fn: Callable[[...], K], **kwargs) -> list[K]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[V]:
        ...  # Don't raise `NotImplementedError` because some pydantic methods may catch that exception.


RequirementProtocol = Union[type[MlflowTaskProtocol], OptionalTask, TaskList]


class TaskImplementationListMeta(Register, Generic[T]):
    def __new__(mcs, classname: str, bases: tuple[type, ...], namespace: dict[str, Any]):
        cls = super(TaskImplementationListMeta, mcs).__new__(mcs, classname, bases, namespace)
        cls.disable_instance_cache()
        return cls

    def __call__(cls, implementations: list[T], *args, **kwargs):
        instance = super(TaskImplementationListMeta, cls).__call__(*args, **kwargs)
        instance.implementations = implementations
        instance.task_id = instance.task_id + "-".join([req.task_id for req in implementations])
        return instance


@dataclass(init=False)
class TaskImplementationList(Generic[T], luigi.Task, metaclass=TaskImplementationListMeta):
    implementations: list[T]

    def requires(self) -> list[T]:
        return self.implementations

    def run(self):
        # Do nothing (`self.requires()` will execute incomplete tasks)
        pass

    def output(self):
        raise NotImplementedError

    def complete(self):
        return all(impl.complete() for impl in self.implementations)

    def apply(self, fn: Callable[[...], K], **kwargs) -> list[K]:
        # Note that `fn` itself is not applied. Only its name is used.
        # So you can pass methods of protocols
        callables: list[Callable] = [getattr(impl, fn.__name__) for impl in self.implementations]
        assert all(callable(maybe_callable) for maybe_callable in callables)
        return [cb(**kwargs) for cb in callables]

    def __hash__(self) -> int:
        return hash(tuple(hash(impl) for impl in self.implementations))

    def __eq__(self, other) -> bool:
        if not isinstance(other, TaskImplementationList):
            return False
        other_impls = other.implementations
        my_impls = self.implementations
        if len(other_impls) != len(my_impls):
            return False
        for a, b in zip(other_impls, my_impls):
            if a != b:
                return False
        return True

    def __len__(self) -> int:
        return len(self.implementations)

    def __iter__(self) -> Iterator[T]:
        return iter(self.implementations)
