import json
from pathlib import Path
from typing import NoReturn, Optional, Protocol, TypedDict, cast, runtime_checkable

import luigi
import pytest

from luigiflow.config import RunnerConfig
from luigiflow.runner import Runner
from luigiflow.task.protocol import MlflowTaskProtocol
from luigiflow.task.task import MlflowTask, TaskConfig
from luigiflow.task.task_types import OptionalTask, TaskList
from luigiflow.task_repository import (
    ProtocolNotRegistered,
    TaskRepository,
    TaskWithTheSameNameAlreadyRegistered,
)
from luigiflow.utils.savers import save_json


@runtime_checkable
class AProtocol(MlflowTaskProtocol, Protocol):
    def do_nothing(self):
        raise NotImplementedError()


@runtime_checkable
class AnotherProtocol(MlflowTaskProtocol, Protocol):
    def method_a(self):
        raise NotImplementedError()


class DoNothingImpl(MlflowTask):
    config = TaskConfig(
        protocols=[
            AProtocol,
        ],
        requirements=dict(),
    )

    def _run(self) -> NoReturn:
        ...

    def do_nothing(self):
        ...


class NewTask(MlflowTask):
    param: str = luigi.Parameter()
    config = TaskConfig(
        protocols=[
            AnotherProtocol,
        ],
        requirements={
            "1": AProtocol,
        },
    )

    def _run(self) -> NoReturn:
        self.save_to_mlflow()

    def method_a(self):
        ...


def test_duplicated_tasks():
    with pytest.raises(TaskWithTheSameNameAlreadyRegistered):
        TaskRepository(
            task_classes=[
                DoNothingImpl,
                DoNothingImpl,
            ],
        )


def test_unknown_protocol():
    class UnknownProtocol(MlflowTaskProtocol):
        ...

    class TaskHavingUnknownProtocol(MlflowTask):
        config = TaskConfig(
            protocols=[
                AProtocol,
            ],
            requirements={
                "unknown": UnknownProtocol,
            },
        )

    repo = TaskRepository(
        task_classes=[
            TaskHavingUnknownProtocol,
        ],
    )
    with pytest.raises(ProtocolNotRegistered):
        repo.generate_task_tree(
            task_params={
                "cls": "TaskHavingUnknownProtocol",
                "params": dict(),
                "requires": {
                    "unknown": {
                        "cls": "UnknownTask",
                        "params": dict(),
                    },
                },
            },
            protocol="DoNothingProtocol",
        )


def test_recursively_nested_task(artifacts_server):
    class TaskA(MlflowTask):
        param: str = luigi.Parameter()
        config = TaskConfig(
            protocols=[
                AProtocol,
            ],
            requirements={
                "req": AProtocol,
            },
            artifact_filenames=dict(),
        )

    class TaskB(MlflowTask):
        config = TaskConfig(
            protocols=[
                AProtocol,
            ],
            requirements={
                "req": AProtocol,
            },
            artifact_filenames=dict(),
        )

    class TaskC(MlflowTask):
        config = TaskConfig(
            protocols=[
                AProtocol,
            ],
            requirements=dict(),
            artifact_filenames=dict(),
        )

    repo = TaskRepository(
        task_classes=[
            TaskA,
            TaskB,
            TaskC,
        ],
    )
    task: MlflowTask = repo.generate_task_tree(
        task_params={
            "cls": "TaskA",
            "params": {
                "param": "top",
            },
            "requires": {
                "req": {
                    "cls": "TaskB",
                    "requires": {
                        "req": {
                            "cls": "TaskA",
                            "params": {
                                "param": "bottom",
                            },
                            "requires": {
                                "req": {
                                    "cls": "TaskB",
                                    "requires": {
                                        "req": {
                                            "cls": "TaskC",
                                        }
                                    },
                                },
                            },
                        }
                    },
                }
            },
        },
        protocol="AProtocol",
    )
    tags = task.to_mlflow_tags_w_parent_tags()
    assert tags["param"] == "top"


def test_too_many_tags(artifacts_server, tmpdir):
    class DoSomething(MlflowTask):
        value: str = luigi.IntParameter(default=1)
        config = TaskConfig(
            protocols=[
                AProtocol,
            ],
            requirements=dict(),
            artifact_filenames=dict(
                out="out.jsonl",
            ),
        )

        def _run(self) -> NoReturn:
            self.save_to_mlflow(
                artifacts_and_save_funcs={
                    "out": (dict(), save_json),
                }
            )

    runner = Runner(
        config=RunnerConfig(
            mlflow_tracking_uri=artifacts_server.url,
            use_local_scheduler=True,
            create_experiment_if_not_existing=True,
        ),
        experiment_repository=TaskRepository(
            task_classes=[DoSomething, NewTask],
        ),
    )
    d = Path(tmpdir.join("sub"))
    d.mkdir()
    config_path = d / "config.jsonnet"
    with open(config_path, "w") as fout:
        json.dump(
            {
                "cls": "NewTask",
                "params": {
                    "param": "test" * 500,  # a long query
                },
                "requires": {
                    "1": {
                        "cls": "DoSomething",
                    }
                },
            },
            fout,
        )
    task, res = runner.run(
        config_jsonnet_path=config_path,
        dry_run=False,
    )
    assert task.complete()


@pytest.mark.parametrize(
    "maybe_task, config, is_ok",
    [
        (
            AProtocol,
            {
                "cls": "TaskB",
                "requires": {
                    "maybe_task": None,
                },
            },
            False,
        ),
        (
            OptionalTask(base_cls=AProtocol),
            {
                "cls": "TaskB",
                "requires": {
                    "maybe_task": None,
                },
            },
            True,
        ),
        (
            AProtocol,
            {
                "cls": "TaskB",
                "requires": {
                    "maybe_task": {
                        "cls": "TaskA",
                    },
                },
            },
            True,
        ),
        (
            OptionalTask(base_cls=AProtocol),
            {
                "cls": "TaskB",
                "requires": {
                    "maybe_task": {
                        "cls": "TaskA",
                    },
                },
            },
            True,
        ),
    ],
)
def test_allow_null_requirements(artifacts_server, tmpdir, maybe_task, config, is_ok):
    class Requirements(TypedDict):
        maybe_task: Optional[AProtocol]

    class TaskA(MlflowTask):
        config = TaskConfig(
            protocols=[AProtocol],
            requirements=dict(),
            artifact_filenames={
                "hi": "hi.json",
            },
        )

        def _run(self) -> NoReturn:
            self.save_to_mlflow(
                artifacts_and_save_funcs={
                    "hi": (dict(), save_json),
                }
            )

    class TaskB(MlflowTask[Requirements]):
        config = TaskConfig(
            protocols=[AnotherProtocol],
            requirements={  # type: ignore
                "maybe_task": maybe_task,
            },
            artifact_filenames={
                "tmp": "tmp.json",
            },
        )

        def _run(self) -> NoReturn:
            self.save_to_mlflow(
                artifacts_and_save_funcs={
                    "tmp": (
                        dict(),
                        save_json,
                    )
                }
            )

    runner = Runner(
        config=RunnerConfig(
            mlflow_tracking_uri=artifacts_server.url,
            use_local_scheduler=True,
            create_experiment_if_not_existing=True,
        ),
        experiment_repository=TaskRepository(
            task_classes=[TaskA, TaskB],
        ),
    )

    d = tmpdir.join("sub")
    d.mkdir()
    config_path = d / "config.jsonnet"
    with open(config_path, "w") as fout:
        json.dump(config, fout)
    if is_ok:
        task, res = runner.run(
            config_jsonnet_path=config_path,
            dry_run=False,
        )
        assert task.complete()
    else:
        with pytest.raises(AssertionError):
            runner.run(
                config_jsonnet_path=config_path,
                dry_run=False,
            )


def test_list_requirements(artifacts_server, tmpdir):
    class GetTextProtocol(MlflowTaskProtocol, Protocol):
        def load_text(self) -> str:
            raise NotImplementedError

    class TaskA(MlflowTask):
        value: int = luigi.IntParameter()
        config = TaskConfig(
            protocols=[GetTextProtocol],
            requirements=dict(),
        )

        def _run(self) -> NoReturn:
            self.save_to_mlflow()

        def load_text(self) -> str:
            return str(self.value)

    class TaskC(MlflowTask):
        value: int = luigi.IntParameter()
        config = TaskConfig(
            protocols=[AnotherProtocol],
            requirements=dict(),
        )

        def _run(self) -> NoReturn:
            self.save_to_mlflow()

    class TaskBRequirements(TypedDict):
        a: TaskList
        a_a: TaskList
        c: AnotherProtocol

    class TaskB(MlflowTask[TaskBRequirements]):
        text: str = luigi.Parameter()
        config = TaskConfig(
            protocols=[AnotherProtocol],
            requirements={
                "a": TaskList(GetTextProtocol),
                "a_a": TaskList(GetTextProtocol),  # test if "a" is not overrode
                "c": AnotherProtocol,
            },
            artifact_filenames={
                "data": "data.json",
            },
        )

        def _run(self) -> NoReturn:
            out: list[str] = self.requires()["a"].apply(GetTextProtocol.load_text)
            aa_out = self.requires()["a_a"].apply(GetTextProtocol.load_text)
            self.save_to_mlflow(
                artifacts_and_save_funcs={
                    "data": ([{"values": out}, {"values_aa": aa_out}], save_json),
                }
            )

    config = """
    {
        cls: "TaskB",
        requires: {
            a: [
                {
                    "cls": "TaskA",
                    "params": {
                        "value": 1,
                    },
                },
                {
                    "cls": "TaskA",
                    "params": {
                        "value": 2,
                    },
                },
            ],
            a_a: [
                {
                    "cls": "TaskA",
                    "params": {
                        "value": 3,
                    },
                },
            ],
            c: {
                cls: "TaskC",
                params: {
                    value: 10,
                }
            },
        },
        params: {
            text: "hello",
        }
    }
    """
    d = tmpdir.join("d")
    d.mkdir()
    config_path = Path(d) / "config.jsonnet"
    with open(config_path, "w") as fout:
        fout.write(config)

    runner = Runner(
        config=RunnerConfig(
            mlflow_tracking_uri=artifacts_server.url,
            use_local_scheduler=True,
            create_experiment_if_not_existing=True,
        ),
        experiment_repository=TaskRepository(
            task_classes=[TaskA, TaskB, TaskC],
        ),
    )
    task, res = runner.run(
        config_jsonnet_path=config_path,
        dry_run=False,
    )
    assert len(task.requires()["a"]) == 2
    assert len(task.requires()["a_a"]) == 1
    assert task.complete()
    with open(task.output()["data"].path) as fin:
        output = json.load(fin)
    assert output == [{"values": ["1", "2"]}, {"values_aa": ["3"]}]
    # check if `__iter__` works
    assert len(list(cast(TaskB, task).requires()["a"])) == 2
