import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, NoReturn, Optional, Tuple, TypeVar, final, Collection, Set

import luigi
import mlflow
from luigi import LocalTarget
from mlflow.entities import Experiment, Run
from mlflow.protos.service_pb2 import ACTIVE_ONLY, RunStatus
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from luigiflow.serializer import MlflowTagSerializer, MlflowTagValue, default_serializer

T = TypeVar("T")


class MlflowTask(luigi.Task):
    """
    This is a luigi's task aiming to save artifacts and/or metrics to an mllfow expriment.
    """

    @classmethod
    def get_tags_to_exclude(cls) -> Set[str]:
        return set()  # default

    @classmethod
    def output_tags_recursively(cls) -> bool:
        return True  # default

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

    @classmethod
    def get_tag_serializer(cls) -> MlflowTagSerializer:
        """
        You normally don't need to override this method.
        :return:
        """
        return default_serializer

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
        By default, this method serialize all the parameters.

        *Difference from metrics*: A metric is the *result* (calculated after the task is complete),
        whereas tags can be determined before running the task.

        :param exclude: Specify parameters not to show in the tags.
        """
        serializer = self.get_tag_serializer()
        return {
            name: serializer.serialize(val)
            for name in self.get_param_names()
            if (
                (val := getattr(self, name)) is not None
                and name not in self.get_tags_to_exclude()
            )
        }

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
        return self._run()

    def search_for_mlflow_run(
        self,
        view_type: RunStatus = ACTIVE_ONLY,
    ) -> Optional[Run]:
        """
        Search an existing run with the same tags.
        """
        experiment: Optional[Experiment] = mlflow.get_experiment_by_name(
            self.get_experiment_name()
        )
        if experiment is None:
            return None
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
            run_view_type=view_type,
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

    def to_mlflow_tags_w_parent_tags(self) -> Dict[str, MlflowTagValue]:
        """
        Serialize tags, including its parents'.
        The format of dict keys is `{param_path}.{param_name}`,
        where `param_path` represents the relative path.
        """
        if not self.output_tags_recursively():
            return self.to_mlflow_tags()

        def to_tags(task: MlflowTask) -> Dict[str, MlflowTagValue]:
            tags = task.to_mlflow_tags()
            if task.requires() is None:
                return tags
            elif len(task.requires()) == 0:
                return tags
            parent_tasks: Dict[str, MlflowTask] = {
                key: val
                for key, val in task.requires().items()
                if isinstance(val, MlflowTask)
            }
            for task_name, t in parent_tasks.items():
                t_tags_w_prefix = {
                    f"{task_name}.{key}": val for key, val in t.to_mlflow_tags_w_parent_tags().items()
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
                        out_path = os.path.join(
                            tmpdir, self.get_artifact_filenames()[name]
                        )
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
