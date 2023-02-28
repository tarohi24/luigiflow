import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Generic,
    Iterator,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from luigi import LocalTarget
from mlflow.entities import Run
from mlflow.protos.service_pb2 import ACTIVE_ONLY, RunStatus
from pydantic import BaseModel, Extra, Field

from luigiflow.domain.serializer import MlflowTagValue, ParameterSerializer
from luigiflow.types import ParameterName, TagValue

_TReq = TypeVar("_TReq", bound=dict)
_K = TypeVar("_K")


@runtime_checkable
class MlflowTaskProtocol(Protocol[_TReq]):
    """
    You can use this protocol to implement task protocols.
    Because a protocol class cannot inherit from non-protocol classes,
    you can use this instead of `MlflowTask`.

    `T` is a `TypedDict` to describe `requires()`.
    """

    def get_parameter_values(self) -> dict[ParameterName, TagValue]:
        """
        Don't name it `get_param_values` because it may conflict with `luigi.Task`.
        :return:
        """
        ...

    @classmethod
    def get_protocols(cls) -> list[type["MlflowTaskProtocol"]]:
        ...

    @classmethod
    def get_experiment_name(cls) -> str:
        ...

    @classmethod
    def get_artifact_filenames(cls) -> dict[str, str]:
        ...

    @classmethod
    def get_tag_serializer(cls) -> ParameterSerializer:
        ...

    # just to note types
    def input(self) -> dict[str, dict[str, LocalTarget]]:
        ...

    def requires(self) -> _TReq:
        ...

    def to_mlflow_tags(self) -> dict[str, MlflowTagValue]:
        ...

    def run(self):
        ...

    def search_for_mlflow_run(
        self, view_type: RunStatus = ACTIVE_ONLY
    ) -> Optional[Run]:
        ...

    def complete(self):
        ...

    def output(self) -> Optional[dict[str, LocalTarget]]:
        ...

    def to_mlflow_tags_w_parent_tags(self) -> dict[str, MlflowTagValue]:
        ...

    def save_to_mlflow(
        self,
        artifacts_and_save_funcs: Optional[
            dict[
                str, Union[Callable[[str], None], tuple[_K, Callable[[_K, str], None]]]
            ]
        ] = None,
        metrics: Optional[dict[str, float]] = None,
        inherit_parent_tags: bool = True,
    ):
        ...

    def logger(self) -> logging.Logger:
        ...

    def get_task_id(self) -> str:
        """
        Return the task ID on Luigi.
        :return:
        """
        ...


_T = TypeVar("_T", bound=MlflowTaskProtocol)
_V = TypeVar("_V", bound=type[MlflowTaskProtocol])
_D = TypeVar("_D", bound=dict)


class TryingToSaveUndefinedArtifact(Exception):
    ...


@dataclass
class OptionalTask(Generic[_V]):
    base_cls: type[_V]

    def __post_init__(self):
        assert issubclass(self.base_cls, MlflowTaskProtocol)


@dataclass
class TaskList(Generic[_V]):
    protocol: type[_V]

    # this method is just to give hints
    def apply(self, fn: Callable[..., _K], **kwargs) -> list[_K]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[_V]:
        if TYPE_CHECKING:
            raise NotImplementedError
        ...  # Don't raise `NotImplementedError` because some pydantic methods may catch that exception.


RequirementProtocol = Union[type[MlflowTaskProtocol], OptionalTask, TaskList]


class TaskConfig(BaseModel, extra=Extra.forbid):
    protocols: list[type[MlflowTaskProtocol]]
    requirements: dict[str, RequirementProtocol] = Field(default_factory=dict)
    artifact_filenames: dict[str, str] = Field(default_factory=dict)
    tags_to_exclude: set[str] = Field(default_factory=set)
