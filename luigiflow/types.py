from typing import TypedDict, Any


# set `total=False` because `requires` is optional
class TaskParameter(TypedDict, total=False):
    cls: str  # either type or class is a reserved word, so I chose to use `cls`
    params: dict[str, Any]
    requires: dict[str, "TaskParameter"]
