import datetime
from typing import Optional, TypedDict, cast

import luigi
import pytest

from luigiflow.infrastructure.luigi.custom_params import (
    OptionalDateParameter,
    OptionalFloatParameter,
    OptionalIntParameter,
    OptionalStrParameter,
)
from luigiflow.domain.serializer import UnknownParameter
from luigiflow.domain.tag_manager import TaskParameter
from luigiflow.domain.task import DeprecatedTaskProtocol, TaskConfig
from luigiflow.infrastructure.luigi.task import DeprecatedTask
from luigiflow.infrastructure.mlflow.collection import TaskCollectionImpl


class TaskProtocol(DeprecatedTaskProtocol):
    ...


class TaskA(DeprecatedTask):
    date_param: datetime.date = luigi.DateParameter(default=datetime.date(2021, 10, 11))
    config = TaskConfig(
        protocols=[
            TaskProtocol,
        ],
        requirements=dict(),
    )


class BRequirements(TypedDict):
    a: TaskProtocol


class TaskB(DeprecatedTask[BRequirements]):
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
        TaskCollectionImpl(
            [
                TaskA,
            ]
        ).generate_task_tree(task_params, DeprecatedTaskProtocol),
    )
    assert task.date_param == datetime.date(2021, 10, 11)
    # test with a custom param
    task_params["params"]["date_param"] = "2021-12-12"
    task = cast(
        TaskA,
        TaskCollectionImpl(
            [
                TaskA,
            ]
        ).generate_task_tree(task_params, DeprecatedTaskProtocol),
    )
    assert task.date_param == datetime.date(2021, 12, 12)
    # test with an invalid param
    task_params["params"]["date_param"] = "invalid"
    with pytest.raises(ValueError):
        TaskCollectionImpl(
            [
                TaskA,
            ]
        ).generate_task_tree(task_params, DeprecatedTaskProtocol)


def test_inconsistent_param_name():
    with pytest.raises(UnknownParameter):
        TaskCollectionImpl([TaskA,]).generate_task_tree(
            task_params={"cls": "TaskA", "params": {"unknown": "hi"}},
            protocol=DeprecatedTaskProtocol,
        )


def test_optional_param():
    class Task(Task):
        maybe_value: Optional[int] = OptionalIntParameter()
        maybe_float: Optional[float] = OptionalFloatParameter()
        maybe_str: Optional[str] = OptionalStrParameter()
        maybe_date: Optional[datetime.date] = OptionalDateParameter()
        maybe_value_default_none: Optional[int] = OptionalIntParameter(default=None)
        config = TaskConfig(
            protocols=[DeprecatedTaskProtocol],
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
    task: Task = TaskCollectionImpl([Task]).generate_task_tree(  # type: ignore
        task_params=config,
        protocol=DeprecatedTaskProtocol,
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
    }
    assert actual == expected

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
    task: Task = TaskCollectionImpl([Task]).generate_task_tree(  # type: ignore
        task_params=config,
        protocol=DeprecatedTaskProtocol,
    )
    assert task.maybe_value == 1
    assert task.maybe_str == "hi"
    assert task.maybe_float == 0.1
    assert task.maybe_date == datetime.date(2020, 1, 1)
    assert task.maybe_value_default_none == 10
