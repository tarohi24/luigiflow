from typing import Any, Protocol, TypedDict, Union

from luigiflow.types import ParameterName, TagKey, TagValue


class TaskTagManager(Protocol):
    def to_mlflow_tags(
        self, param_values: dict[ParameterName, TagValue]
    ) -> dict[TagKey, TagValue]:
        raise NotImplementedError


class TaskParameter(TypedDict, total=False):
    cls: str  # either type or class is a reserved word, so I chose to use `cls`
    params: dict[str, Any]
    # ignore type errors. mypy doesn't support recursive types
    requires: dict[str, Union["TaskParameter", list["TaskParameter"]]]  # type: ignore
