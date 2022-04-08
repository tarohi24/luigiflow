from pathlib import Path
from pathlib import Path
from typing import Any, Union

from pydantic import BaseModel, HttpUrl, Field


class RunnerConfig(BaseModel):
    mlflow_tracking_uri: str
    use_local_scheduler: bool = True
    create_experiment_if_not_existing: bool = False
    luigi_build_kwargs: dict[str, Any] = Field(default_factory=dict)
