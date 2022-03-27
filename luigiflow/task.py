import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, NoReturn, Optional, TypeVar, final, Any, Protocol

import luigi
import mlflow
from luigi import LocalTarget
from luigi.task_register import Register
from mlflow.entities import Experiment, Run
from mlflow.protos.service_pb2 import ACTIVE_ONLY, RunStatus
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from luigiflow.serializer import MlflowTagSerializer, MlflowTagValue, default_serializer

T = TypeVar("T")


class TaskConfig(BaseModel):
    experiment_name: str
    sub_experiment_name: str
    protocols: list[type[Protocol]]
    tags_to_exclude: set[str] = Field(default_factory=set)
    output_tags_recursively: bool = Field(default=True)


class MlflowTaskMeta(Register):

    def __new__(mcs, classname: str, bases: tuple[type, ...], namespace: dict[str, Any]):
        cls = super(MlflowTaskMeta, mcs).__new__(mcs, classname, bases, namespace)
        try:
            config: TaskConfig = namespace["config"]
        except KeyError:
            raise ValueError(f"{classname} doesn't have a Config.")
        cls.experiment_name = config.experiment_name
        cls.sub_experiment_name = config.sub_experiment_name
        cls.protocols = config.protocols
        # check types
        for prt in cls.protocols:
            if not issubclass(cls, prt):
                raise ValueError(f"{cls} is not a {prt}")
        cls.tags_to_exclude = config.tags_to_exclude
        cls.output_tags_recursively = config.output_tags_recursively
        return cls


class MlflowTask(luigi.Task, metaclass=MlflowTaskMeta):
    """
    This is a luigi's task aiming to save artifacts and/or metrics to an mllfow expriment.
    """
    config = TaskConfig(
        experiment_name="",  # dummy
        sub_experiment_name="",  # dummy
        protocols=[],
    )

    @classmethod
    @final
    def get_protocols(cls) -> list[Protocol]:
        return cls.protocols

    @classmethod
    @final
    def get_tags_to_exclude(cls) -> set[str]:
        return cls.tags_to_exclude

    @classmethod
    @final
    def get_experiment_name(cls) -> str:
        """
        :return: name of the mlflow experiment corresponding to this task.
        """
        return cls.experiment_name

    @classmethod
    @final
    def get_subtask_name(cls) -> str:
        return cls.sub_experiment_name

    @classmethod
    def get_artifact_filenames(cls) -> dict[str, str]:
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
    def input(self) -> dict[str, dict[str, LocalTarget]]:
        return super(MlflowTask, self).input()

    def requires(self) -> dict[str, luigi.Task]:
        """
        :return: A dictionary consisting of {task_name: task}
        """
        raise NotImplementedError()

    def to_mlflow_tags(self) -> dict[str, MlflowTagValue]:
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
                and name not in self.tags_to_exclude
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

    def output(self) -> Optional[dict[str, LocalTarget]]:
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

    def to_mlflow_tags_w_parent_tags(self) -> dict[str, MlflowTagValue]:
        """
        Serialize tags, including its parents'.
        The format of dict keys is `{param_path}.{param_name}`,
        where `param_path` represents the relative path.
        """
        if not self.output_tags_recursively:
            return self.to_mlflow_tags()

        def to_tags(task: MlflowTask) -> dict[str, MlflowTagValue]:
            tags = task.to_mlflow_tags()
            if task.requires() is None:
                return tags
            elif len(task.requires()) == 0:
                return tags
            parent_tasks: dict[str, MlflowTask] = {
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
        artifacts_and_save_funcs: dict[str, tuple[T, Callable[[T, str], None]]] = None,
        metrics: dict[str, float] = None,
        inherit_parent_tags: bool = True,
    ):
        """
        Register artifacts and/or metrics to mlflow.
        """
        artifact_paths: list[str] = []
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
