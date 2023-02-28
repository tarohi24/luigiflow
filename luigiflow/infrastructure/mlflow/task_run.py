import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union, cast

import _jsonnet
import luigi
import mlflow

from luigiflow.config import RunnerConfig
from luigiflow.domain.collection import TaskCollection
from luigiflow.domain.tag_param import TaskParameter
from luigiflow.domain.task_run import (
    InvalidJsonnetFileError,
    K,
    TaskRun,
    TaskRunRepository,
)
from luigiflow.infrastructure.luigi.task import MlflowTask
from luigiflow.types import RunReturn, TagKey, TagValue, TaskClassName


def _load_task_params(path: Path) -> TaskParameter:
    try:
        json_str = _jsonnet.evaluate_file(str(path))
    except RuntimeError as e:
        raise InvalidJsonnetFileError(str(e))
    data = cast(TaskParameter, json.loads(json_str))
    return data


@dataclass
class MlflowTaskRunRepository(TaskRunRepository):
    config: RunnerConfig
    experiment_repository: TaskCollection

    def run_with_task_param(
        self,
        task_param: TaskParameter,
        dry_run: bool = False,
    ) -> RunReturn:
        task = self.experiment_repository.generate_task_tree(
            task_params=task_param,
        )
        experiment_name = task.get_experiment_name()
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        if mlflow.get_experiment_by_name(experiment_name) is None:
            if self.config.create_experiment_if_not_existing:
                mlflow.create_experiment(experiment_name)
            else:
                raise ValueError(
                    f"Experiment {experiment_name} not found at {self.config.mlflow_tracking_uri}"
                )
        if dry_run:
            return task, None
        res = luigi.build(
            [
                task,
            ],
            local_scheduler=self.config.use_local_scheduler,
            detailed_summary=True,
            **self.config.luigi_build_kwargs,
        )
        return task, res

    def run(
        self,
        config_jsonnet_path: Path,
        dry_run: bool = False,
    ) -> RunReturn:
        task_param = _load_task_params(path=config_jsonnet_path)
        return self.run_with_task_param(
            task_param=task_param,
            dry_run=dry_run,
        )

    def search_for_runs(
        self, task_name: TaskClassName, tags: dict[TagKey, TagValue]
    ) -> Optional[TaskRun]:
        pass

    def save_run(
        self,
        task: MlflowTask,
        artifacts_and_save_funcs: Optional[
            dict[str, Union[Callable[[str], None], tuple[K, Callable[[K, str], None]]]]
        ] = None,
        metrics: Optional[dict[str, float]] = None,
        inherit_parent_tags: bool = True,
    ) -> TaskRun:
        pass
