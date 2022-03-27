import json
import pickle
from typing import Any, Union

import pandas as pd


def save_dataframe(
    df: pd.DataFrame,
    path: str,
    save_index: bool = False,
):
    df.to_csv(path, index=save_index)


def save_pickle(obj: Any, path: str):
    with open(path, "wb") as fout:
        pickle.dump(obj, fout)


def save_json(
    json_data: Union[list[dict[str, Any]], dict[str, Any]],
    path: str,
    indent: int = 4,
    ensure_ascii: bool = False,
):
    assert isinstance(json_data, list) or isinstance(json_data, dict)
    with open(path, "w") as fout:
        json.dump(json_data, fout, indent=indent, ensure_ascii=ensure_ascii)
    return
