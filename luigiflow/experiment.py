from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow

from luigiflow.config.jsonnet import JsonnetConfigLoader
from luigiflow.config.run import RunnerConfig
from luigiflow.task import MlflowTask


@dataclass
class Experiment:
    name: str
    task_classes: dict[str, type[MlflowTask]]

    def get_task_class(self, name: str) -> type[MlflowTask]:
        return self.task_classes[name]
