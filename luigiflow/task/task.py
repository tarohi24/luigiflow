import hashlib
import logging
import os
import tempfile
from operator import itemgetter
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    NoReturn,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
    final,
)

import luigi
import mlflow
from luigi import LocalTarget
from luigi.task_register import Register
from mlflow.entities import Experiment, Run, RunInfo
from mlflow.protos.service_pb2 import ACTIVE_ONLY, RunStatus
from pydantic import BaseModel, Extra, Field
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from luigiflow.serializer import MlflowTagSerializer, MlflowTagValue, default_serializer
from luigiflow.task.protocol import MlflowTaskProtocol
from luigiflow.task.task_types import (
    OptionalTask,
    RequirementProtocol,
    TaskImplementationList,
    TaskList,
)

T = TypeVar("T", bound=MlflowTaskProtocol)
_TReq = TypeVar("_TReq", bound=dict)
K = TypeVar("K")


class TryingToSaveUndefinedArtifact(Exception):
    ...


class TaskConfig(BaseModel, extra=Extra.forbid):
    protocols: list[type[MlflowTaskProtocol]]
    requirements: dict[str, RequirementProtocol] = Field(default_factory=dict)
    artifact_filenames: dict[str, str] = Field(default_factory=dict)
    tags_to_exclude: set[str] = Field(default_factory=set)


class MlflowTaskMeta(Register, Generic[_TReq], type(Protocol)):  # type: ignore[misc]
    def __new__(
        mcs, classname: str, bases: tuple[type, ...], namespace: dict[str, Any]
    ):
        cls = super(MlflowTaskMeta, mcs).__new__(mcs, classname, bases, namespace)
        try:
            config: TaskConfig = namespace["config"]
        except KeyError:
            raise ValueError(f"{classname} doesn't have a Config.")
        cls.experiment_name = cls.__name__
        cls.protocols = config.protocols
        cls.requirements = dict()
        cls.requirements_required = dict()
        for key, maybe_req_type in config.requirements.items():
            if isinstance(maybe_req_type, OptionalTask):
                cls.requirements[key] = maybe_req_type.base_cls
                cls.requirements_required[key] = False
            elif isinstance(maybe_req_type, TaskList):
                cls.requirements[key] = maybe_req_type
                cls.requirements_required[key] = True
            else:
                # not generics, i.e., should be `MlflowTaskProtocol`
                cls.requirements[key] = maybe_req_type
                cls.requirements_required[key] = True

        cls.tags_to_exclude = config.tags_to_exclude
        cls.artifact_filenames = config.artifact_filenames
        cls.param_types = {
            key: type(maybe_param)
            for key, maybe_param in namespace.items()
            if isinstance(maybe_param, luigi.Parameter)
        }
        cls.disable_instance_cache()
        return cls

    # `T` is an `MlflowTask` class
    def __call__(cls, requirements_impl: _TReq, *args, **kwargs):
        """
        This specifies how to instantiate `MlflowTask`, i.e. this is `Mlflow.__init__`.
        Because `luigi.Task` has a metaclass `Register`, you cannot override `luigi.Task.__init__`.
        :param requirements_impl:
        :param args:
        :param kwargs:
        :return:
        """
        instance = super(MlflowTaskMeta, cls).__call__(*args, **kwargs)
        instance.requirements_impl = dict()
        for key, maybe_impl in requirements_impl.items():
            if maybe_impl is None:
                assert not cls.requirements_required[
                    key
                ], f"{key} requires requirements"
                instance.requirements_impl[key] = None
            elif isinstance(maybe_impl, list):
                # list of classes
                assert isinstance(cls.requirements[key], TaskList)
                instance.requirements_impl[key] = [impl for impl in maybe_impl]
            else:
                instance.requirements_impl[key] = maybe_impl
        # renew task_id to distinguish tasks with the same params but different requirements
        instance.task_id += "-".join(
            [
                req.task_id
                for req in instance.requirements_impl.values()
                if req is not None
            ]
        )
        return instance


class MlflowTask(luigi.Task, MlflowTaskProtocol[_TReq], metaclass=MlflowTaskMeta):
    """
    This is a luigi's task aiming to save artifacts and/or metrics to an mllfow expriment.
    """

    config = TaskConfig(
        protocols=[],
        requirements=dict(),
    )

    @classmethod
    @final
    def get_protocols(cls) -> list[type["MlflowTaskProtocol"]]:
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
    def get_artifact_filenames(cls) -> dict[str, str]:
        """
        :return: `{file_id: filename w/ an extension}`. `file_id` is an ID used only in this task.
        """
        return cls.artifact_filenames

    @classmethod
    def get_tag_serializer(cls) -> MlflowTagSerializer:
        """
        You normally don't need to override this method.
        :return:
        """
        return default_serializer

    # just to note types
    def input(self) -> dict[str, dict[str, LocalTarget]]:
        return super(MlflowTask, self).input()  # type: ignore[safe-super]

    @final
    def requires(self) -> _TReq:
        """
        :return: A dictionary consisting of {task_name: task}
        """
        return self.requirements_impl

    def to_mlflow_tags(self) -> dict[str, MlflowTagValue]:
        """
        Serialize parameters of this task.
        By default, this method serialize all the parameters.

        *Difference from metrics*: A metric is the *result* (calculated after the task is complete),
        whereas tags can be determined before running the task.

        :param exclude: Specify parameters not to show in the tags.
        """
        serializer = self.get_tag_serializer()
        base = {
            name: serializer.serialize(getattr(self, name))
            for name in self.get_param_names()
            if name not in self.tags_to_exclude
        }
        base["name"] = str(self.__class__.__name__)
        return base

    def _run(self) -> None:
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
        # query_items = [f'tag.{pname} = "{pval}"' for pname, pval in ]
        # query = " and ".join(query_items)
        df = mlflow.search_runs(
            experiment_ids=[
                experiment.experiment_id,
            ],
            run_view_type=view_type,
            output_format="pandas",
        )
        for tag_name, pval in self.to_mlflow_tags_w_parent_tags().items():
            pname = f"tags.{tag_name}"
            if pname not in df.columns:
                return None
            df = df[df[pname] == str(pval)]
        if len(df) > 0:
            assert len(df) == 1
            row = df.to_dict("records")[0]
            return Run(  # note that this convert is not complete
                run_info=RunInfo(
                    run_uuid=row["run_id"],
                    experiment_id=row["experiment_id"],
                    user_id=row["tags.mlflow.user"],
                    status=row["status"],
                    start_time=row["start_time"],
                    end_time=row["end_time"],
                    lifecycle_stage="",
                    artifact_uri=row["artifact_uri"],
                    run_id=row["run_id"],
                ),
                run_data=None,
            )
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

        def to_tags(task: MlflowTask) -> dict[str, MlflowTagValue]:
            tags = task.to_mlflow_tags()
            maybe_requirements: Optional[
                dict[str, MlflowTaskProtocol]
            ] = task.requires()
            if maybe_requirements is None:
                return tags
            else:
                if len(maybe_requirements) == 0:
                    return tags
            parent_tasks: dict[str, MlflowTask | TaskImplementationList] = {
                key: val
                for key, val in maybe_requirements.items()
                if (
                    isinstance(val, MlflowTask)
                    or isinstance(val, TaskImplementationList)
                )
            }
            for task_name, t in parent_tasks.items():
                if isinstance(t, MlflowTask):
                    parent_tags: dict[
                        str, MlflowTagValue
                    ] = t.to_mlflow_tags_w_parent_tags()
                    if len(parent_tags) > 100:
                        t_tags_w_prefix = {f"{task_name}_hash": parent_tags["_hash"]}
                    else:
                        t_tags_w_prefix = {
                            f"{task_name}.{key}": val
                            for key, val in parent_tags.items()
                        }
                elif isinstance(t, TaskImplementationList):
                    if len(t) > 100:
                        m = hashlib.md5()
                        for task in t:
                            task = cast(MlflowTask, task)
                            hash_val = str(task.to_mlflow_tags_w_parent_tags()["_hash"])
                            m.update(hash_val.encode("utf-8"))
                        t_tags_w_prefix = {f"{task_name}_hash": m.hexdigest()}
                    else:
                        t_tags_w_prefix = dict()
                        for i, task in enumerate(t):
                            ctags = task.to_mlflow_tags_w_parent_tags()
                            if len(ctags) > 100:
                                t_tags_w_prefix = t_tags_w_prefix | {
                                    f"{task_name}.{i}_hash": ctags["_hash"]
                                }
                            else:
                                t_tags_w_prefix = t_tags_w_prefix | {
                                    f"{task_name}.{i}.{key}": val
                                    for key, val in ctags.items()
                                }
                else:
                    raise AssertionError()

                tags = dict(**tags, **t_tags_w_prefix)

            return tags

        serialized_tags = to_tags(self)
        tags_sorted: list[tuple[str, str]] = sorted(list(serialized_tags.items()), key=itemgetter(0))  # type: ignore
        m = hashlib.md5()
        for key, val in tags_sorted:
            m.update(f"{key}={val}".encode("utf-8"))
        serialized_tags["_hash"] = m.hexdigest()
        return serialized_tags

    @final
    def save_to_mlflow(
        self,
        artifacts_and_save_funcs: Optional[
            dict[str, Union[Callable[[str], None], tuple[K, Callable[[K, str], None]]]]
        ] = None,
        metrics: Optional[dict[str, float]] = None,
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
                    for name, tup_or_fn in artifacts_and_save_funcs.items():
                        try:
                            output_file_name = self.get_artifact_filenames()[name]
                        except KeyError:
                            raise TryingToSaveUndefinedArtifact(f"Unknown file: {name}")
                        out_path = os.path.join(tmpdir, output_file_name)
                        self.logger.info(f"Saving artifact to {out_path}")
                        if isinstance(tup_or_fn, tuple):
                            artifact, save_fn = tup_or_fn
                            save_fn(artifact, out_path)
                        else:
                            assert callable(tup_or_fn)
                            tup_or_fn(out_path)
                        artifact_paths.append(out_path)
                    for path in artifact_paths:
                        mlflow.log_artifact(path)
            # Save tags
            tags_dict: dict[str, Any] = (
                self.to_mlflow_tags_w_parent_tags()
                if inherit_parent_tags
                else self.to_mlflow_tags()
            )
            tags: list[tuple[str, Any]] = list(
                [(key, val) for key, val in tags_dict.items()]
            )
            n_tags = len(tags)
            start_pos = list(range(0, n_tags, 50))
            end_pos = start_pos[1:] + [n_tags]
            for pos, next_pos in zip(start_pos, end_pos):
                mlflow.set_tags(dict(tags[pos:next_pos]))
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

    def get_task_id(self) -> str:
        return self.task_id
