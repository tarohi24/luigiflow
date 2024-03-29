import logging
from typing import Callable, Optional, Protocol, TypeVar, Union, runtime_checkable

from luigi import LocalTarget
from mlflow.entities import Run
from mlflow.protos.service_pb2 import ACTIVE_ONLY, RunStatus

from luigiflow.serializer import MlflowTagSerializer, MlflowTagValue

_TReq = TypeVar(
    "_TReq", bound=dict, covariant=True
)  # to denote the type of `task.requires()`
K = TypeVar("K")  # for `save_artifacts`


@runtime_checkable
class MlflowTaskProtocol(Protocol[_TReq]):
    """
    You can use this protocol to implement task protocols.
    Because a protocol class cannot inherit from non-protocol classes,
    you can use this instead of `MlflowTask`.

    `T` is a `TypedDict` to describe `requires()`.
    """

    @classmethod
    def get_protocols(cls) -> list[type["MlflowTaskProtocol"]]:
        ...

    @classmethod
    def get_tags_to_exclude(cls) -> set[str]:
        ...

    @classmethod
    def get_experiment_name(cls) -> str:
        ...

    @classmethod
    def get_artifact_filenames(cls) -> dict[str, str]:
        ...

    @classmethod
    def get_tag_serializer(cls) -> MlflowTagSerializer:
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
            dict[str, Union[Callable[[str], None], tuple[K, Callable[[K, str], None]]]]
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
