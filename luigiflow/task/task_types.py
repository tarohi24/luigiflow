from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterator,
    Protocol,
    TypeVar,
    Union,
)

import luigi
from luigi.task_register import Register

from luigiflow.task.protocol import MlflowTaskProtocol

V = TypeVar("V", bound=type[MlflowTaskProtocol])
K = TypeVar("K")
T = TypeVar("T", bound=dict)  # to denote the type of `task.requires()`
_TT = TypeVar("_TT", bound=MlflowTaskProtocol)


@dataclass
class OptionalTask(Generic[V]):
    base_cls: type[V]

    def __post_init__(self):
        assert issubclass(self.base_cls, MlflowTaskProtocol)


@dataclass
class TaskList(Generic[V]):
    protocol: type[V]

    # this method is just to give hints
    def apply(self, fn: Callable[..., K], **kwargs) -> list[K]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[V]:
        if TYPE_CHECKING:
            raise NotImplementedError
        ...  # Don't raise `NotImplementedError` because some pydantic methods may catch that exception.


RequirementProtocol = Union[type[MlflowTaskProtocol], OptionalTask, TaskList]


class TaskImplementationListMeta(Register, Generic[_TT]):
    def __new__(
        mcs, classname: str, bases: tuple[type, ...], namespace: dict[str, Any]
    ):
        cls = super(TaskImplementationListMeta, mcs).__new__(
            mcs, classname, bases, namespace
        )
        cls.disable_instance_cache()
        return cls

    def __call__(cls, implementations: list[_TT], *args, **kwargs):
        instance = super(TaskImplementationListMeta, cls).__call__(*args, **kwargs)
        instance.implementations = implementations
        instance.task_id = instance.task_id + "-".join(
            [req.get_task_id() for req in implementations]
        )
        return instance


@dataclass(init=False)
class TaskImplementationList(
    Generic[_TT], luigi.Task, metaclass=TaskImplementationListMeta
):
    implementations: list[_TT]

    def requires(self) -> list[_TT]:
        return self.implementations

    def run(self):
        # Do nothing (`self.requires()` will execute incomplete tasks)
        pass

    def output(self):
        raise NotImplementedError

    def complete(self):
        return all(impl.complete() for impl in self.implementations)

    def apply(self, fn: Callable[..., K], **kwargs) -> list[K]:
        # Note that `fn` itself is not applied. Only its name is used.
        # So you can pass methods of protocols
        callables: list[Callable] = [
            getattr(impl, fn.__name__) for impl in self.implementations
        ]
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

    def __iter__(self) -> Iterator[_TT]:
        return iter(self.implementations)
