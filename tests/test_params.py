import datetime
from typing import TypedDict

import luigi

from luigiflow.task import MlflowTask, TaskConfig, MlflowTaskProtocol


class TaskProtocol(MlflowTaskProtocol):
    ...


class TaskA(MlflowTask):
    config = TaskConfig(
        experiment_name="hi",
        protocols=[TaskProtocol, ],
        requirements=dict(),
    )


class BRequirements(TypedDict):
    a: TaskProtocol


class TaskB(MlflowTask[BRequirements]):
    date_param: datetime.date = luigi.DateParameter()
    config = TaskConfig(
        experiment_name="hi",
        protocols=[TaskProtocol, ],
        requirements={
            "a": TaskProtocol,
        },
    )


def test_serialize_date_param():
    task_params = {
        "cls": "TaskB",
        "params": {
            "date_param": "2021-10-12",
        },
        "requirements": {
            "a": {
                "cls": "TaskA",
            }
        }
    }

    config_path = tmpdir.mkdir("sub").join("config.jsonnet")
    with config_path.open("w") as fout:
        fout.write('''
                local val = 3.0;
                {
                    type: "TaskB",
                    params: {
                        date_start: std.extVar("DATE_START"),
                    },
                    requires: {
                        a: {
                            type: "TaskA",
                            params: {
                                value: val,
                            },
                        }
                    }
                }
                ''')
    invalid_params = [
        {"DATE": "2021-11-11"},
        {"DATE_START": "2021-11-11"},
    ]
    with pytest.raises(InvalidJsonnetFileError):
        runner.run("SaveJson", config_path)

    # valid keys, invalid values
    invalid_params = [
        {"DATE_START": "hi!"},  # invalid
        {"DATE_START": "2021-11-11"},  # valid
    ]
    with pytest.raises(ValueError):
        runner.run("SaveJson", config_path)

    valid_params = [
        {"DATE_START": "2021-11-12"},  # valid
        {"DATE_START": "2021-11-11"},  # valid
    ]
    task, res = runner.run("SaveJson", config_path)
    assert len(task) == 2
    assert res.status == LuigiStatusCode.SUCCESS
    # Check if all the tasks ran
    for param in valid_params:
        config_loader = JsonnetConfigLoader(external_variables=param)
        with config_loader.load(runner.config.config_path):
            assert TaskA().complete()
