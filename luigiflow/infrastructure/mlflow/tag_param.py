import hashlib
from dataclasses import dataclass, field
from typing import Optional, cast

from luigi import Parameter

from luigiflow.domain.serializer import ParameterSerializer, MlflowTagValue
from luigiflow.domain.tag_manager import TaskTagManager
from luigiflow.domain.task import DeprecatedTaskProtocol
from luigiflow.infrastructure.luigi.task import DeprecatedTask, TaskImplementationList
from luigiflow.types import ParameterName, TagKey, TagValue, TaskClassName



def _generate_tag_dict_for_task_list(task_list: TaskImplementationList, task_name: str) -> str:
    if len(task_list.implementations) > 100:
        m = hashlib.md5()
        for task in task_list.implementations:
            task = cast(DeprecatedTask, task)
            hash_val = str(task.to_mlflow_tags_w_parent_tags()["_hash"])
            m.update(hash_val.encode("utf-8"))
            t_tags_w_prefix = {f"{task_name}_hash": m.hexdigest()}
    else:
        t_tags_w_prefix = dict()
        for i, task in enumerate(t):
            ctags = task.to_mlflow_tags_w_parent_tags()
            if len(ctags) > 100:
                t_tags_w_prefix = t_tags_w_prefix | {
                    f"{task_name}.{i}_hash": ctags["_hash"]
                }
            else:
                t_tags_w_prefix = t_tags_w_prefix | {
                    f"{task_name}.{i}.{key}": val
                    for key, val in ctags.items()
                }
        return t_tags_w_prefix

def _to_tags_recursively(task: DeprecatedTaskProtocol) -> dict[str, MlflowTagValue]:
    # because this is an implementation specific for Mlflow, you can assert that the implementation of
    # `MlflowTaskProtocol` is `MlflowTask`
    tags = task.to_mlflow_tags()
    maybe_requirements: Optional[
        dict[str, DeprecatedTaskProtocol]
    ] = task.requires()
    if maybe_requirements is None:
        return tags
    else:
        if len(maybe_requirements) == 0:
            return tags
    parent_tasks: dict[str, DeprecatedTask | TaskImplementationList] = {
        key: val
        for key, val in maybe_requirements.items()
        if (
                isinstance(val, DeprecatedTask)
                or isinstance(val, TaskImplementationList)
        )
    }
    for task_name, t in parent_tasks.items():
        if isinstance(t, DeprecatedTask):
            parent_tags: dict[
                str, MlflowTagValue
            ] = t.to_mlflow_tags_w_parent_tags()
            if len(parent_tags) > 100:
                t_tags_w_prefix = {f"{task_name}_hash": parent_tags["_hash"]}
            else:
                t_tags_w_prefix = {
                    f"{task_name}.{key}": val
                    for key, val in parent_tags.items()
                }
        elif isinstance(t, TaskImplementationList):
            t_tags_w_prefix = _generate_tag_dict_for_task_list(task_list=t, task_name=task_name)

        else:
            raise AssertionError()

        tags = dict(**tags, **t_tags_w_prefix)

    return tags


@dataclass
class MlflowTagManager(TaskTagManager):
    task_name: TaskClassName
    params: dict[ParameterName, Parameter]
    serializer: ParameterSerializer
    param_names_to_exclude_from_tags: set[ParameterName] = field(default_factory=set)

    def get_param_names_not_to_tag(self) -> set[ParameterName]:
        return self.param_names_to_exclude_from_tags

    @staticmethod
    def param_name_to_tag_key(param_name: ParameterName) -> TagKey:
        return param_name

    def _to_mlflow_tags(self, param_values: dict[ParameterName, TagValue]):
        assert (
            set(param_values.keys()) == set(self.params.keys()),
            f"Expected {self.params.keys()}, but got {param_values.keys()}",
        )
        base = {
            self.param_name_to_tag_key(param_name=param_name): self.serializer.serialize(param)
            for param_name, param in param_values.items()
            if param_name not in self.param_names_to_exclude_from_tags
        }
        base["name"] = str(self.task_name)
        return base

    def to_tags(
        self,
        task: DeprecatedTaskProtocol,
        include_parent_tags: bool,
    ) -> dict[TagKey, TagValue]:
        if not include_parent_tags:
            return self._to_mlflow_tags(param_values=task.get_parameter_values())
