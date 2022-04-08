import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import _jsonnet
import luigi
import mlflow
from luigi.execution_summary import LuigiRunResult

from luigiflow.config.run import RunnerConfig
from luigiflow.task_repository import TaskRepository
from luigiflow.types import TaskParameter

RunReturn = tuple[luigi.Task, Optional[LuigiRunResult]]


class InvalidJsonnetFileError(Exception):
    ...


def _load_task_params(path: Path) -> TaskParameter:
    try:
        json_str = _jsonnet.evaluate_file(str(path))
    except RuntimeError as e:
        raise InvalidJsonnetFileError(str(e))
    data = cast(TaskParameter, json.loads(json_str))
    return data


@dataclass
class Runner:
    config: RunnerConfig
    experiment_repository: TaskRepository

    def run(
        self,
        protocol_name: str,
        config_jsonnet_path: Path,
        dry_run: bool = False,
    ) -> RunReturn:
        task_params = _load_task_params(path=config_jsonnet_path)
        task = self.experiment_repository.generate_task_tree(
            protocol_name=protocol_name,
            task_params=task_params,
        )
        experiment_name = task.get_experiment_name()
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        if mlflow.get_experiment_by_name(experiment_name) is None:
            if self.config.create_experiment_if_not_existing:
                mlflow.create_experiment(experiment_name)
            else:
                raise ValueError(f"Experiment {experiment_name} not found at {self.config.mlflow_tracking_uri}")
        if dry_run:
            return task, None
        res = luigi.build(
            [task, ],
            local_scheduler=self.config.use_local_scheduler,
            detailed_summary=True,
            **self.config.luigi_build_kwargs
        )
        return task, res
