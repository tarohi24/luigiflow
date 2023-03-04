import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
    final,
)

import luigi
import mlflow
from _operator import itemgetter
from luigi import LocalTarget
from luigi.task_register import Register
from mlflow.entities import Experiment, Run, RunInfo
from mlflow.protos.service_pb2 import ACTIVE_ONLY, RunStatus
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from luigiflow.domain.serializer import MlflowTagValue, default_serializer
from luigiflow.domain.task import (
    DeprecatedTaskProtocol,
    OptionalTask,
    TaskConfig,
    TaskList,
    TryingToSaveUndefinedArtifact,
)
from luigiflow.types import TagValue, ParameterName

_TReq = TypeVar("_TReq", bound=dict)
_K = TypeVar("_K")
_TT = TypeVar("_TT", bound=DeprecatedTaskProtocol)


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

        cls.artifact_filenames = config.artifact_filenames
        cls.tag_manager = TaskTagm  (
            task_name=classname,
            params={
                name: type(maybe_param)
                for name, maybe_param in namespace.items()
                if isinstance(maybe_param, luigi.Parameter)
            },
            serializer=default_serializer,
            param_names_to_exclude_from_tags=config.tags_to_exclude,
        )
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


class DeprecatedTask(luigi.Task, DeprecatedTaskProtocol[_TReq], metaclass=MlflowTaskMeta):
    """
    This is a luigi's task aiming to save artifacts and/or metrics to an mllfow expriment.
    """

    config = TaskConfig(
        protocols=[],
        requirements=dict(),
    )

    def get_parameter_values(self) -> dict[ParameterName, TagValue]:
        return {name: getattr(self, name) for name in self.get_param_names()}

    @classmethod
    @final
    def get_protocols(cls) -> list[type["DeprecatedTaskProtocol"]]:
        return cls.protocols

    @classmethod
    @final
    def get_experiment_name(cls) -> str:
        """
        I'm not sure if this class is responsible for managing its experiment name.
        """
        return cls.experiment_name

    @classmethod
    @final
    def get_artifact_filenames(cls) -> dict[str, str]:
        """
        :return: `{file_id: filename w/ an extension}`. `file_id` is an ID used only in this task.
        """
        return cls.artifact_filenames

    # just to note types
    def input(self) -> dict[str, dict[str, LocalTarget]]:
        return super(DeprecatedTask, self).input()  # type: ignore[safe-super]

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
        return self.tag_manager.to_tags(task=self, include_parent_tags=False)

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
            dict[
                str, Union[Callable[[str], None], tuple[_K, Callable[[_K, str], None]]]
            ]
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


class TaskImplementationListMeta(Register, Generic[_TT]):
    def __new__(
        mcs, classname: str, bases: tuple[type, ...], namespace: dict[str, Any]
    ):
        cls = super(TaskImplementationListMeta, mcs).__new__(
            mcs, classname, bases, namespace
        )
        cls.disable_instance_cache()
        return cls

    def __call__(cls, implementations: list[_TT], *args, **kwargs):
        instance = super(TaskImplementationListMeta, cls).__call__(*args, **kwargs)
        instance.implementations = implementations
        instance.task_id = instance.task_id + "-".join(
            [req.get_task_id() for req in implementations]
        )
        return instance


@dataclass(init=False)
class TaskImplementationList(
    Generic[_TT], luigi.Task, metaclass=TaskImplementationListMeta
):
    implementations: list[_TT]

    def requires(self) -> list[_TT]:
        return self.implementations



    def run(self):
        # Do nothing (`self.requires()` will execute incomplete tasks)
        pass

    def output(self):
        raise NotImplementedError

    def complete(self):
        return all(impl.complete() for impl in self.implementations)

    def apply(self, fn: Callable[..., _K], **kwargs) -> list[_K]:
        # Note that `fn` itself is not applied. Only its name is used.
        # So you can pass methods of protocols
        callables: list[Callable] = [
            getattr(impl, fn.__name__) for impl in self.implementations
        ]
        assert all(callable(maybe_callable) for maybe_callable in callables)
        return [cb(**kwargs) for cb in callables]

    def __hash__(self) -> int:
        return hash(tuple(hash(impl) for impl in self.implementations))

    def __eq__(self, other) -> bool:
        if not isinstance(other, TaskImplementationList):
            return False
        other_impls = other.implementations
        my_impls = self.implementations
        if len(other_impls) != len(my_impls):
            return False
        for a, b in zip(other_impls, my_impls):
            if a != b:
                return False
        return True

    def __len__(self) -> int:
        return len(self.implementations)

    def __iter__(self) -> Iterator[_TT]:
        return iter(self.implementations)
