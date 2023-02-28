from dataclasses import dataclass, field
from typing import Any, TypedDict, Union

from luigi import Parameter

from luigiflow.domain.serializer import ParameterSerializer
from luigiflow.types import ParameterName, TagKey, TagValue, TaskClassName


@dataclass
class TagManager:
    task_name: TaskClassName
    params: dict[ParameterName, Parameter]
    serializer: ParameterSerializer
    param_names_to_exclude_from_tags: set[ParameterName] = field(default_factory=set)

    def get_param_names_not_to_tag(self) -> set[ParameterName]:
        return self.param_names_to_exclude_from_tags

    @staticmethod
    def param_name_to_tag_key(param_name: ParameterName) -> TagKey:
        return param_name

    def to_mlflow_tags(
        self, param_values: dict[ParameterName, TagValue]
    ) -> dict[TagKey, TagValue]:
        assert (
            set(param_values.keys()) == set(self.params.keys()),
            f"Expected {self.params.keys()}, but got {param_values.keys()}",
        )
        base = {
            self.param_name_to_tag_key(
                param_name=param_name
            ): self.serializer.serialize(param)
            for param_name, param in param_values.items()
            if param_name not in self.param_names_to_exclude_from_tags
        }
        base["name"] = str(self.task_name)
        return base


class TaskParameter(TypedDict, total=False):
    cls: str  # either type or class is a reserved word, so I chose to use `cls`
    params: dict[str, Any]
    # ignore type errors. mypy doesn't support recursive types
    requires: dict[str, Union["TaskParameter", list["TaskParameter"]]]  # type: ignore
