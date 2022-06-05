import datetime
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, Union, cast, Optional

MlflowTagValue = Union[str, int, float]
T = TypeVar("T")


@dataclass
class MlflowTagSerializer:
    # T isn't V because V is a value type, whereas T is a parameter
    # `serializers` is not a dict because the order matters.
    # e.g. a `datetime` is also a `date`. Then it has to prioritize the `datetime` serializer.
    serializers: list[tuple[type[T], Callable[[T], MlflowTagValue]]]

    def serialize(self, val: Any) -> MlflowTagValue:
        if val is None:
            return "null"
        val_type: type = type(val)
        for typ, fn in self.serializers:
            if isinstance(val, typ):
                return fn(val)
        raise TypeError(f"Got an unknown parameter type: {val_type}")


def identical_function(val: T) -> T:
    return val


default_serializer: MlflowTagSerializer = MlflowTagSerializer(
    [
        (str, identical_function),
        (int, identical_function),
        (float, identical_function),
        (bool, lambda b: int(b)),
        (datetime.date, lambda d: cast(datetime.date, d).isoformat()),
    ]
)


def get_optional_serializer(serializer: Callable[[str], T]) -> Callable[[str], Optional[T]]:
    def fn(s: str) -> Optional[T]:
        if s == "null":
            return None
        elif s == "None":
            return None
        else:
            return serializer(s)

    return fn


DESERIALIZERS: dict[str, Callable[[str], Any]] = {
    "Parameter": str,
    "IntParameter": int,
    "BoolParameter": (
        lambda s: {
            "True": True,
            "1": True,
            "true": True,
            "False": False,
            "0": False,
            "false": False,
        }[s]
    ),
    "FloatParameter": float,
    "DateParameter": datetime.date.fromisoformat,
    "OptionalIntParameter": get_optional_serializer(int),
    "OptionalFloatParameter": get_optional_serializer(float),
    "OptionalStrParameter": get_optional_serializer(str),
    "OptionalDateParameter": get_optional_serializer(datetime.date.fromisoformat),
}
