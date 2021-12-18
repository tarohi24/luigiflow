import datetime
from dataclasses import dataclass
from typing import Generic, TypeVar, Callable, Union, Dict, Type, Any, cast

import luigi

MlflowTagValue = Union[str, int, float]
T = TypeVar("T")


@dataclass
class MlflowTagSerializer:
    # T isn't V because V is a value type, whereas T is a parameter
    serializers: Dict[Type[T], Callable[[T], MlflowTagValue]]

    def serialize(self, val: Any) -> MlflowTagValue:
        val_type: Type = type(val)
        try:
            fn = self.serializers[val_type]
        except KeyError:
            raise TypeError(f"Got an unknown parameter type: {val_type}")
        return fn(val)


def identical_function(val: T) -> T:
    return val


default_serializer: MlflowTagSerializer = MlflowTagSerializer(
    {
        str: identical_function,
        int: identical_function,
        bool: lambda b: int(b),
        datetime.date: lambda d: cast(datetime.date, d).isoformat(),
    }
)