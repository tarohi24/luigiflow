from typing import NoReturn, Dict

import luigi
import pandas as pd

from luigiflow.config import JsonnetConfigLoader
from luigiflow.savers import save_dataframe
from luigiflow.task import MlflowTask


class SomeTask(MlflowTask):
    int_param: int = luigi.IntParameter()
    str_param: str = luigi.Parameter()

    @classmethod
    def get_experiment_name(cls) -> str:
        return "example"

    @classmethod
    def get_artifact_filenames(cls) -> Dict[str, str]:
        return {
            "data": "data.csv",
        }

    def requires(self) -> Dict[str, luigi.Task]:
        return dict()

    def _run(self) -> NoReturn:
        df = pd.DataFrame([
            {"label": "A", "score": 1.1},
            {"label": "B", "score": 0.2},
        ])
        self.save_to_mlflow(
            artifacts_and_save_funcs={
                "data": (df, save_dataframe),
            }
        )


def test_load_config(project_root, tmpdir):
    tmpfile = tmpdir.mkdir("sub").join("config.jsonnet")
    with tmpfile.open("w") as fout:
        fout.write("""
        local int_param = 3;
        {
            "SomeTask": {
                "int_param": int_param,
                "str_param": std.extVar("EXT_VAR"),
            },
        }
        """)
    loader = JsonnetConfigLoader(
        external_variables={"EXT_VAR": "hi!"}
    )
    with loader.load(tmpfile):
        task = SomeTask()
        assert task.int_param == 3
        assert task.str_param == "hi!"


def test_prioritize_jsonnet(tmpdir):
    directory = tmpdir.mkdir("sub")
    config_path = directory.join("config.jsonnet")
    with config_path.open("w") as fout:
        fout.write("""
        local int_param = 3;
        {
            "SomeTask": {
                "int_param": int_param,
                "str_param": std.extVar("EXT_VAR"),
            },
        }
        """)
    with directory.join("luigi.cfg").open("w") as fout:
        fout.write("""
        [SomeTask]
        int_param = 1
        """)
    loader = JsonnetConfigLoader(
        external_variables={"EXT_VAR": "hi!"}
    )
    with loader.load(config_path):
        task = SomeTask()
        assert task.int_param == 3
        assert task.str_param == "hi!"


def test_load_from_dict():
    config = {
        "SomeTask": {
            "int_param": 2,
            "str_param": "hi",
        },
    }
    loader = JsonnetConfigLoader()
    with loader.load_from_dict(config):
        task = SomeTask()
        assert task.int_param == 2
        assert task.str_param == "hi"


def test_load_integer_extvar(tmpdir):
    ext_vars = {
        "value": 1,
    }
    config_path = tmpdir.mkdir("sub").join("conifg.jsonnet")
    with config_path.open("w") as fout:
        fout.write("""
    local value = std.extVar("value");
    {
        "SomeTask": {
            "str_param": "hi",
            "int_param": value,
        }
    }
    """)
    with JsonnetConfigLoader(external_variables=ext_vars).load(config_path):
        task = SomeTask()
        assert task.int_param == 1
