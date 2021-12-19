import pickle
from typing import Any

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
