from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import luigi
import mlflow
from luigi.execution_summary import LuigiRunResult

from luigiflow.config.run import RunnerConfig
from luigiflow.task_repository import TaskRepository

RunReturn = tuple[list[luigi.Task], Optional[LuigiRunResult]]


@dataclass
class Runner:
    config: RunnerConfig
    experiment_repository: TaskRepository

    def run(
        self,
        protocol_name: str,
        external_params: list[dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> RunReturn:
        external_params = external_params or []
        tasks = self.experiment_repository.generate_tasks(
            protocol_name=protocol_name,
            external_params=external_params,
            context_config_path=self.config.config_path,
        )
        assert Path(self.config.config_path).exists()
        experiment_name = tasks[0].get_experiment_name()
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
