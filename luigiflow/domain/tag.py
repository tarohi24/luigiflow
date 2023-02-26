from dataclasses import dataclass, field

from luigi import Parameter

from luigiflow.domain.serializer import ParameterSerializer
from luigiflow.domain.types import TaskClassName, ParameterName, TagKey, TagValue


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

    def to_mlflow_tags(self) -> dict[TagKey, TagValue]:
        base = {
            self.param_name_to_tag_key(param_name=param_name): self.serializer.serialize(param)
            for param_name, param in self.params
            if param_name not in self.param_names_to_exclude_from_tags
        }
        base["name"] = str(self.task_name)
        return base
