import datetime
from typing import TypedDict, cast, Optional

import luigi
import pytest

from luigiflow.custom_params import (
    OptionalIntParameter,
    OptionalStrParameter,
    OptionalDateParameter,
    OptionalFloatParameter,
)
from luigiflow.task import MlflowTask, TaskConfig, MlflowTaskProtocol
from luigiflow.task_repository import TaskRepository, UnknownParameter
from luigiflow.types import TaskParameter
from luigiflow.utils.testing import assert_two_tags_equal_wo_hashes


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


def test_optional_param():
    class Task(MlflowTask):
        maybe_value: Optional[int] = OptionalIntParameter()
        maybe_float: Optional[float] = OptionalFloatParameter()
        maybe_str: Optional[str] = OptionalStrParameter()
        maybe_date: Optional[datetime.date] = OptionalDateParameter()
        maybe_value_default_none: Optional[int] = OptionalIntParameter(default=None)
        config = TaskConfig(
            protocols=[TaskProtocol],
        )

    config = {
        "cls": "Task",
        "params": {
            "maybe_value": None,
            "maybe_float": None,
            "maybe_str": None,
            "maybe_date": None,
        },
    }
    task: Task = TaskRepository([Task]).generate_task_tree(  # type: ignore
        task_params=config,
        protocol=TaskProtocol,
    )
    assert task.maybe_value is None
    assert task.maybe_str is None
    assert task.maybe_date is None
    actual = task.to_mlflow_tags()
    expected = {
        "maybe_value": "null",
        "maybe_float": "null",
        "maybe_date": "null",
        "maybe_str": "null",
        "maybe_value_default_none": "null",
        "name": "Task",
        "_hash": "...",
    }
    assert_two_tags_equal_wo_hashes(actual, expected)

    config = {
        "cls": "Task",
        "params": {
            "maybe_value": 1,
            "maybe_str": "hi",
            "maybe_float": 0.1,
            "maybe_date": "2020-01-01",
            "maybe_value_default_none": 10,
        },
    }
    task: Task = TaskRepository([Task]).generate_task_tree(  # type: ignore
        task_params=config,
        protocol=TaskProtocol,
    )
    assert task.maybe_value == 1
    assert task.maybe_str == "hi"
    assert task.maybe_float == 0.1
    assert task.maybe_date == datetime.date(2020, 1, 1)
    assert task.maybe_value_default_none == 10
