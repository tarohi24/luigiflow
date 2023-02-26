from dataclasses import dataclass

from luigi import Parameter

from luigiflow.domain.serializer import ParameterSerializer


@dataclass
class TagManager:
    specs: dict[str, Parameter]
    serializer: ParameterSerializer

    ...