from typing import TypeAlias, Union

from pydantic import constr

TaskClassName: TypeAlias = str  # e.g., `ExportTagsToCSV`
ParameterName: TypeAlias = str  # e.g., `threshold`
TagKey: TypeAlias = str  # e.g., `threshold`
TagValue: TypeAlias = Union[str, int, float]  # e.g., `0.5`
