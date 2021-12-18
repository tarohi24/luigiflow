import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, final, TypeVar, Union, NoReturn, Tuple, Callable, List

import luigi
import mlflow
from luigi import LocalTarget
from mlflow.entities import Run
from mlflow.protos.service_pb2 import ACTIVE_ONLY
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

T = TypeVar("T")
MlflowTagValue = Union[str, int, float]


class MlflowTask(luigi.Task):
    """
    This is a luigi's task aiming to save artifacts and/or metrics to an mllfow expriment.
    """

    @classmethod
    def get_experiment_name(cls) -> str:
        """
        :return: name of the mlflow experiment corresponding to this task.
        """
        raise NotImplementedError()

    @classmethod
    def get_artifact_filenames(cls) -> Dict[str, str]:
        """
        :return: `{file_id: filename w/ an extension}`. `file_id` is an ID used only in this task.
        """
        raise NotImplementedError()

    # just to note types
    def input(self) -> Dict[str, Dict[str, LocalTarget]]:
        return super(MlflowTask, self).input()

    def requires(self) -> Dict[str, luigi.Task]:
        """
        :return: A dictionary consisting of {task_name: task}
        """
        raise NotImplementedError()

    def to_mlflow_tags(self) -> Dict[str, MlflowTagValue]:
        """
        Serialize parameters of this task.

        *Difference from metrics*: A metric is the *result* (calculated after the task is complete),
        whereas tags can be determined before running the task.
        """
        raise NotImplementedError()

    def _run(self) -> NoReturn:
        """
        Specify a flow to save artifacts / metrics.
        """
        raise NotImplementedError()

    @final
    def run(self):
        self.logger.info(f"Start {self.__class__.__name__}")
        mlflow.set_experiment(self.get_experiment_name())
        self.logger.info("Initialize done")
        self._run()

    def search_for_mlflow_run(self) -> Optional[Run]:
        """
        Search an existing run with the same tags.
        """
        experiment = mlflow.get_experiment_by_name(self.get_experiment_name())
        query_items = [
            f'tag.{pname} = "{pval}"'
            for pname, pval in self.to_mlflow_tags_w_parent_tags().items()
        ]
        query = " and ".join(query_items)
        res = mlflow.search_runs(
            experiment_ids=[
                experiment.experiment_id,
            ],
            filter_string=query,
            max_results=1,
            run_view_type=ACTIVE_ONLY,
            output_format="list",
        )
        if len(res) > 0:
            return res[0]
        else:
            return None

    def complete(self):
        """
        A task's completion is determined by the existence of a run with the sampe tags in mlflow
        """
        is_complete = self.search_for_mlflow_run() is not None
        self.logger.info(f"is_complete: {is_complete}")
        return is_complete

    def output(self) -> Optional[Dict[str, LocalTarget]]:
        """
        :return: A dict consisting of artifact names and their paths.
        If this task isn't completed, this method returns None.
        """
        maybe_mlf_run = self.search_for_mlflow_run()
        if maybe_mlf_run is None:
            return None
        paths = {
            key: Path(maybe_mlf_run.info.artifact_uri) / fname
            for key, fname in self.get_artifact_filenames().items()
        }
        # logging
        return {key: LocalTarget(str(p)) for key, p in paths.items()}

    @final
    def to_mlflow_tags_w_parent_tags(self) -> Dict[str, MlflowTagValue]:
        """
        Serialize tags, including its parents'.
        The format of dict keys is `{param_path}.{param_name}`,
        where `param_path` represents the relative path.
        e.g. Let this task have a parameter called `param_1`
        and two requirements whose names are `aaa` and `bbb`, respectively.
        Let `aaa` have the parameters `X` and `Y`,
        and `bbb` have no parameters but a required task `ccc`, which has a parameter called `Z`.
        Then this method returns the following tags.

        ```python
        >>> class TaskA(MlflowTask):
        >>>     param_1: int = luigi.IntParameter(default=10)
        >>>     def requires(self): return dict()
        >>>     def to_mlflow_tags(self): return {"param_1": self.param_1}
        >>>     def _run(self): ...
        >>>
        >>> class TaskB(MlflowTask):
        >>>     message: str = luigi.Parameter(default='hi')
        >>>     def requires(self): return dict()
        >>>     def to_mlflow_tags(self): return {"message": self.message}
        >>>     def _run(self): ...
        >>>
        >>> class TaskC(MlflowTask):
        >>>     def requires(self): return {"bbb": TaskB()}
        >>>     def to_mlflow_tags(self): return dict()
        >>>     def _run(self): ...
        >>>
        >>> class TaskD(MlflowTask):
        >>>     threshold: float = luigi.FloatParameter(default=2e+3)
        >>>     def requires(self): return {'aaa': TaskA(), 'ccc': TaskC()}
        >>>     def to_mlflow_tags(self): return {"threshold": self.threshold}
        >>>     def _run(self): ...
        >>>
        >>> task = TaskD()
        >>> task.to_mlflow_tags_w_parent_tags()
        {
            "aaa.param_1": 100,
            "ccc.bbb.message": "hi",
            "threshold": 2e+3,
        }
        ```
        """

        def to_tags(task: MlflowTask) -> Dict[str, MlflowTagValue]:
            tags = task.to_mlflow_tags()
            if task.requires() is None:
                return tags
            parent_tasks: Dict[str, MlflowTask] = {
                key: val
                for key, val in task.requires().items()
                if isinstance(val, MlflowTask)
            }
            for task_name, t in parent_tasks.items():
                t_tags_w_prefix = {
                    f"{task_name}.{key}": val for key, val in to_tags(t).items()
                }
                tags = dict(**tags, **t_tags_w_prefix)
            return tags

        return to_tags(self)

    @final
    def save_to_mlflow(
        self,
        artifacts_and_save_funcs: Dict[str, Tuple[T, Callable[[T, str], None]]] = None,
        metrics: Dict[str, float] = None,
        inherit_parent_tags: bool = True,
    ):
        """
        Register artifacts and/or metrics to mlflow.
        """
        artifact_paths: List[str] = []
        artifacts_and_save_funcs = artifacts_and_save_funcs or dict()
        with mlflow.start_run():
            # Save artifacts
            if len(artifacts_and_save_funcs) > 0:
                # Save artifacts
                with tempfile.TemporaryDirectory() as tmpdir:
                    for name, (artifact, save_fn) in artifacts_and_save_funcs.items():
                        out_path = os.path.join(tmpdir, self.get_artifact_filenames()[name])
                        self.logger.info(f"Saving artifact to {out_path}")
                        save_fn(artifact, out_path)
                        artifact_paths.append(out_path)
                    for path in artifact_paths:
                        mlflow.log_artifact(path)
            # Save tags
            mlflow.set_tags(
                (
                    self.to_mlflow_tags_w_parent_tags()
                    if inherit_parent_tags
                    else self.to_mlflow_tags()
                )
            )
            # Save metrics
            if metrics is not None:
                mlflow.log_metrics(metrics)

    @property
    def logger(self):
        return logging.getLogger(self.get_experiment_name())

    def enable_tqdm(self):
        tqdm.pandas()
        return logging_redirect_tqdm(
            loggers=[
                self.logger,
            ]
        )
