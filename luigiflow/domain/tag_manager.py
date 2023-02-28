from typing import Any, Protocol, TypedDict, Union

from luigiflow.domain.task import MlflowTaskProtocol
from luigiflow.types import ParameterName, TagKey, TagValue


class TaskTagManager(Protocol):
    def to_tags(
        self, task: MlflowTaskProtocol, include_parent_tags: bool,
    ) -> dict[TagKey, TagValue]:
        raise NotImplementedError


class TaskParameter(TypedDict, total=False):
    cls: str  # either type or class is a reserved word, so I chose to use `cls`
    params: dict[str, Any]
    # ignore type errors. mypy doesn't support recursive types
    requires: dict[str, Union["TaskParameter", list["TaskParameter"]]]  # type: ignore
