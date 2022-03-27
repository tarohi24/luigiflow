from dataclasses import dataclass

from luigiflow.task import MlflowTask


@dataclass
class Experiment:
    name: str
    task_classes: dict[str, type[MlflowTask]]

    def get_task_class(self, name: str) -> type[MlflowTask]:
        return self.task_classes[name]
