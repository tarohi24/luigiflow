from pathlib import Path
from pathlib import Path
from typing import Any, Union

from pydantic import BaseModel, HttpUrl, Field


class RunnerConfig(BaseModel):
    mlflow_tracking_uri: HttpUrl
    config_path: Union[str, Path]  # Don't use `PathLike` because `BaseModel` doesn't have a validator for it
    use_local_scheduler: bool = True
    create_experiment_if_not_existing: bool = False
    luigi_build_kwargs: dict[str, Any] = Field(default_factory=dict)
