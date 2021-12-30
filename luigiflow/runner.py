from os import PathLike
from pathlib import Path
from typing import Optional, Dict, Any, Type

import luigi
import mlflow
from luigi.execution_summary import LuigiRunResult

from luigiflow.config import JsonnetConfigLoader
from luigiflow.task import MlflowTask


def run(
    task_cls: Type[MlflowTask],
    mlflow_tracking_uri: str,
    config_path: PathLike,
    local_scheduler: bool = True,
    create_experiment_if_not_existing: bool = False,
    luigi_build_kwargs: Optional[Dict[str, Any]] = None,
) -> LuigiRunResult:
    assert Path(config_path).exists()
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    experiment_name: str = task_cls.get_experiment_name()
    if mlflow.get_experiment_by_name(experiment_name) is None:
        if create_experiment_if_not_existing:
            mlflow.create_experiment(experiment_name)
        else:
            raise ValueError()  # TODO: error message

    config_loader = JsonnetConfigLoader()
    with config_loader.load(config_path):
        task = task_cls()
        res = luigi.build(
            [task, ],
            local_scheduler=local_scheduler,
            detailed_summary=True,
            **(luigi_build_kwargs or dict())
        )
    return res
