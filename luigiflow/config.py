import json
import tempfile
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Dict, Any

import _jsonnet
import toml
from luigi.configuration import add_config_path
from luigi.configuration.base_parser import BaseParser

JsonDict = Dict[str, Any]


class InvalidJsonnetFileError(Exception):
    pass


@dataclass
class ConfigContext:
    data: JsonDict
    tmpfile_suffix: str = field(default=".toml")
    tmp_toml_path: Path = field(init=False)
    orig_conf: BaseParser = field(init=False)

    def __enter__(self):
        """
        See the background of this implementation here:
        https://github.com/spotify/luigi/blob/4d0576c7c265afcb228097af79f316ba0de0242c/test/helpers.py#L57
        :return:
        """
        fp = tempfile.NamedTemporaryFile(suffix=self.tmpfile_suffix, mode='w')
        self.tmp_toml_path = Path(fp.name)
        with self.tmp_toml_path.open('w') as fout:
            toml.dump(self.data, fout)
        add_config_path(str(self.tmp_toml_path))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tmp_toml_path.unlink(missing_ok=True)

    def get_dependencies(self, key: str = 'dependencies') -> Dict[str, str]:
        return self.data.get(key, dict())


@dataclass
class JsonnetConfigLoader:
    external_variables: Dict[str, Any] = field(
        default_factory=lambda: dict(),
    )  # Its type is equivalent to `JsonDict`, but semantically different.

    def load(self, path: PathLike) -> ConfigContext:
        try:
            json_str = _jsonnet.evaluate_file(
                str(path),
                ext_vars=self.external_variables
            )
        except RuntimeError as e:
            raise InvalidJsonnetFileError(str(e))
        param_dict = json.loads(json_str)
        return self.load_from_dict(param_dict)

    @staticmethod
    def load_from_dict(param_dict: Dict[str, Any]):
        """
        I prefer to use this method rather than calling `ConfigContext` directly
        because I can hide `ConcigContext` from outside this module.

        :param param_dict:
        :return:
        """
        config_context = ConfigContext(data=param_dict)
        return config_context
