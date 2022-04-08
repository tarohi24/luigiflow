from typing import TypedDict, Any


class TaskParameter(TypedDict):
    cls: str  # either type or class is a reserved word, so I chose to use `cls`
    params: dict[str, Any]
    requires: dict[str, "TaskParameter"]
