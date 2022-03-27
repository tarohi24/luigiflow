from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Optional, Any

import luigi
import mlflow
from luigi.execution_summary import LuigiRunResult

from luigiflow.config.run import RunnerConfig
from luigiflow.experiment_repository import ExperimentRepository
from luigiflow.task import MlflowTask

RunReturn = tuple[list[luigi.Task], Optional[LuigiRunResult]]


@dataclass
class Runner:
    config: RunnerConfig
    experiment_repository: ExperimentRepository

    def run(
        self,
        experiment_name: str,
        experiment_sub_name: str,
        params: list[dict[str, Any]],
        dry_run: bool = False,
    ) -> RunReturn:
        tasks = self.experiment_repository.generate_tasks(
            name=experiment_name,
            sub_name=experiment_sub_name,
            params=params,
            context_config_path=self.config.config_path,
        )
        assert Path(self.config.config_path).exists()
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        if mlflow.get_experiment_by_name(experiment_name) is None:
            if self.config.create_experiment_if_not_existing:
                mlflow.create_experiment(experiment_name)
            else:
                raise ValueError(f"Experiment {experiment_name} not found at {self.config.mlflow_tracking_uri}")
        if dry_run:
            return tasks, None
        res = luigi.build(
            tasks,
            local_scheduler=self.config.use_local_scheduler,
            detailed_summary=True,
            **self.config.luigi_build_kwargs
        )
        return tasks, res


def run_multiple_tasks_of_single_task_cls(
    task_cls: type[MlflowTask],
    params: list[dict[str, Any]],
    mlflow_tracking_uri: str,
    config_path: PathLike,
    local_scheduler: bool = True,
    create_experiment_if_not_existing: bool = False,
    luigi_build_kwargs: Optional[dict[str, Any]] = None,
    dry_run: bool = False,
) -> RunReturn:
    ...


def run(
    task_cls: type[MlflowTask],
    **kwargs
) -> RunReturn:
    """
    Run a task without ext params
    :param task_cls:
    :param kwargs:
    :return:
    """
    return run_multiple_tasks_of_single_task_cls(
        task_cls=task_cls,
        params=list(),
        **kwargs
    )
