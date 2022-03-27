from dataclasses import dataclass
from pathlib import Path
from typing import Any

from luigiflow.config.jsonnet import JsonnetConfigLoader
from luigiflow.experiment import Experiment
from luigiflow.task import MlflowTask


@dataclass
class ExperimentRepository:
    experiments: dict[str, Experiment]

    def generate_tasks(
        self,
        name: str,
        sub_name: str,
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
                    task = experiment.get_task_class(sub_name)()
                    tasks.append(task)
        else:
            # no external params
            config_loader = JsonnetConfigLoader()
            with config_loader.load(context_config_path):
                task = experiment.get_task_class(sub_name)()
                tasks.append(task)
        return tasks
