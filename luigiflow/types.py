from typing import Any, TypedDict, Union


# set `total=False` because `requires` is optional
class TaskParameter(TypedDict, total=False):
    cls: str  # either type or class is a reserved word, so I chose to use `cls`
    params: dict[str, Any]
    # ignore type errors. mypy doesn't support recursive types
    requires: dict[str, Union["TaskParameter", list["TaskParameter"]]]  # type: ignore
