from typing import Optional, TypeAlias, Union

import luigi
from luigi.execution_summary import LuigiRunResult
from pydantic import constr

TaskClassName: TypeAlias = str  # e.g., `ExportTagsToCSV`
ParameterName: TypeAlias = str  # e.g., `threshold`
TagKey: TypeAlias = str  # e.g., `threshold`
TagValue: TypeAlias = Union[str, int, float]  # e.g., `0.5`
ArtifactURI: TypeAlias = str
RunReturn = tuple[luigi.Task, Optional[LuigiRunResult]]
