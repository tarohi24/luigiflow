import datetime
from typing import TypedDict, cast

import luigi
import pytest

from luigiflow.task import MlflowTask, TaskConfig, MlflowTaskProtocol
from luigiflow.task_repository import TaskRepository, UnknownParameter
from luigiflow.types import TaskParameter


class TaskProtocol(MlflowTaskProtocol):
    ...


class TaskA(MlflowTask):
    date_param: datetime.date = luigi.DateParameter(default=datetime.date(2021, 10, 11))
    config = TaskConfig(
        protocols=[
            TaskProtocol,
        ],
        requirements=dict(),
    )


class BRequirements(TypedDict):
    a: TaskProtocol


class TaskB(MlflowTask[BRequirements]):
    date_param: datetime.date = luigi.DateParameter()
    config = TaskConfig(
        protocols=[
            TaskProtocol,
        ],
        requirements={
            "a": TaskProtocol,
        },
    )


def test_serialize_date_param():
    task_params: TaskParameter = {
        "cls": "TaskA",
        "params": dict(),
    }
    # test default value
    task = cast(
        TaskA,
        TaskRepository(
            [
                TaskA,
            ]
        ).generate_task_tree(task_params, TaskProtocol),
    )
    assert task.date_param == datetime.date(2021, 10, 11)
    # test with a custom param
    task_params["params"]["date_param"] = "2021-12-12"
    task = cast(
        TaskA,
        TaskRepository(
            [
                TaskA,
            ]
        ).generate_task_tree(task_params, TaskProtocol),
    )
    assert task.date_param == datetime.date(2021, 12, 12)
    # test with an invalid param
    task_params["params"]["date_param"] = "invalid"
    with pytest.raises(ValueError):
        TaskRepository(
            [
                TaskA,
            ]
        ).generate_task_tree(task_params, TaskProtocol)


def test_inconsistent_param_name():
    with pytest.raises(UnknownParameter):
        TaskRepository([TaskA,]).generate_task_tree(
            task_params={"cls": "TaskA", "params": {"unknown": "hi"}},
            protocol=TaskProtocol,
        )
