import datetime
from typing import cast

import luigi
import pytest

from luigiflow.domain.task import DeprecatedTaskProtocol, TaskConfig
from luigiflow.infrastructure.luigi.task import DeprecatedTask
from luigiflow.infrastructure.mlflow.collection import TaskCollectionImpl


class DummyProtocolDeprecated(DeprecatedTaskProtocol):
    ...


class Task(DeprecatedTask):
    param_int: int = luigi.IntParameter(default=10)
    param_str: str = luigi.Parameter(default="hi")
    param_bool: str = luigi.BoolParameter(default=True)
    param_date: datetime.date = luigi.DateParameter(default=datetime.date(2021, 1, 2))
    param_large_value: float = luigi.FloatParameter(default=2e11)
    config = TaskConfig(
        protocols=[
            DummyProtocolDeprecated,
        ],
        requirements=dict(),
        tags_to_exclude={"param_int", "param_date", "param_large_value"},
    )


class AnotherTask(Task):
    strange_param = luigi.Parameter(default=Task)  # invalid value
    config = TaskConfig(
        protocols=[
            DummyProtocolDeprecated,
        ],
        requirements=dict(),
        artifact_filenames=dict(),
    )

    def do_nothing(self):
        ...


class TestToTags:
    @pytest.fixture()
    def task(self) -> Task:
        repo = TaskCollectionImpl([DeprecatedTask])
        task = repo.generate_task_tree(
            task_params={
                "cls": "Task",
                "params": dict(),
            },
            protocol="DummyProtocol",
        )
        return cast(DeprecatedTask, task)

    def test_simply_to_tags(self, task):
        actual = task.to_mlflow_tags()
        expected = {
            "name": "Task",
            "param_str": "hi",
            "param_bool": 1,
        }
        assert actual == expected

    def test_if_tags_to_exclude_work(self, task):
        # disable `tags_to_exclude`
        task.tag_manager.param_names_to_exclude_from_tags = set()
        actual = task.to_mlflow_tags()
        expected = {
            "name": "Task",
            "param_int": 10,
            "param_str": "hi",
            "param_bool": 1,
            "param_date": "2021-01-02",
            "param_large_value": 200000000000.0,
        }
        assert actual == expected

    def test_invalid_param_type(self):
        repo = TaskCollectionImpl([DeprecatedTask])
        repo._protocols["DummyProtocol"].register(AnotherTask)
        task = repo.generate_task_tree(
            task_params={
                "cls": "AnotherTask",
                "params": dict(),
            },
            protocol="DummyProtocol",
        )

        with pytest.raises(TypeError):
            task.to_mlflow_tags()
