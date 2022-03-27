from dataclasses import dataclass
from pathlib import Path
from typing import Any

from luigiflow.config.jsonnet import JsonnetConfigLoader
from luigiflow.experiment import Experiment
from luigiflow.task import MlflowTask


@dataclass
class ExperimentsRepository:
    experiments: dict[str, Experiment]
    dependencies: dict[str, str]

    def __post_init__(self):
        experiment_names = set(self.experiments.keys())
        dependency_names = set(self.dependencies.keys())
        if len(experiment_names - dependency_names) > 0:
            raise ValueError(f"Dependencies not set: {experiment_names - dependency_names}")
        if len(dependency_names - experiment_names):
            raise ValueError(f"Unknown dependencies: {dependency_names - experiment_names}")

    def generate_tasks(
        self,
        name: str,
        params: list[dict[str, Any]],
        context_config_path: Path,
    ) -> list[MlflowTask]:
        if not isinstance(params, list):
            raise ValueError("Pass a list of kwargs `params`")
        experiment = self.experiments[name]
        tasks = []
        if len(params) > 0:
            for param in params:
                config_loader = JsonnetConfigLoader(external_variables=param)
                with config_loader.load(context_config_path):
                    task = experiment.get_task_class(self.dependencies[name])()
                    tasks.append(task)
        else:
            # no external params
            config_loader = JsonnetConfigLoader()
            with config_loader.load(context_config_path):
                task = experiment.get_task_class(self.dependencies[name])()
                tasks.append(task)
        return tasks
