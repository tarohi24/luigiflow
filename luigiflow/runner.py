from os import PathLike
from pathlib import Path
from typing import Optional, Dict, Any, Type, List

import luigi
import mlflow

from luigiflow.config import JsonnetConfigLoader
from luigiflow.task import MlflowTask


def run_multiple_tasks_of_single_task_cls(
    task_cls: Type[MlflowTask],
    params: List[Dict[str, Any]],
    mlflow_tracking_uri: str,
    config_path: PathLike,
    local_scheduler: bool = True,
    create_experiment_if_not_existing: bool = False,
    luigi_build_kwargs: Optional[Dict[str, Any]] = None,
):
    assert Path(config_path).exists()
    luigi_build_kwargs = luigi_build_kwargs or dict()
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    experiment_name: str = task_cls.get_experiment_name()
    if mlflow.get_experiment_by_name(experiment_name) is None:
        if create_experiment_if_not_existing:
            mlflow.create_experiment(experiment_name)
        else:
            raise ValueError()  # TODO: error message

    tasks = []
    for param in params:
        config_loader = JsonnetConfigLoader(external_variables=param)
        with config_loader.load(config_path):
            task = task_cls()
            tasks.append(task)
    res = luigi.build(
        tasks,
        local_scheduler=local_scheduler,
        detailed_summary=True,
        **luigi_build_kwargs
    )
    return res


def run(
    task_cls: Type[MlflowTask],
    **kwargs
):
    return run_multiple_tasks_of_single_task_cls(
        task_cls=task_cls,
        params=list(),
        **kwargs
    )