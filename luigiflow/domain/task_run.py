from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol, TypeVar, Union
from uuid import UUID

from luigiflow.domain.tag_param import TaskParameter
from luigiflow.infrastructure.luigi.task import MlflowTask
from luigiflow.types import ArtifactURI, RunReturn, TagKey, TagValue, TaskClassName

K = TypeVar("K")


@dataclass
class TaskRun:
    uuid: UUID
    task_name: TaskClassName
    artifact_uri: ArtifactURI


class InvalidJsonnetFileError(Exception):
    ...


class TaskRunRepository(Protocol):
    def search_for_runs(
        self,
        task_name: TaskClassName,
        tags: dict[TagKey, TagValue],
    ) -> Optional[TaskRun]:
        raise NotImplementedError

    def save_run(
        self,
        task: MlflowTask,
        artifacts_and_save_funcs: Optional[
            dict[str, Union[Callable[[str], None], tuple[K, Callable[[K, str], None]]]]
        ] = None,
        metrics: Optional[dict[str, float]] = None,
        inherit_parent_tags: bool = True,
    ) -> TaskRun:
        raise NotImplementedError

    def run_with_task_param(
        self,
        task_param: TaskParameter,
        dry_run: bool = False,
    ) -> RunReturn:
        raise NotImplementedError

    def run(
        self,
        config_jsonnet_path: Path,
        dry_run: bool = False,
    ) -> RunReturn:
        raise NotImplementedError
