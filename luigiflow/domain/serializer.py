import datetime
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar, Union, cast

from luigi import Parameter

MlflowTagValue = Union[str, int, float]
T = TypeVar("T")


class UnknownParameter(Exception):
    ...


@dataclass
class _Serializer(Generic[T]):
    typ: type[T]
    serialize_fn: Callable[[T], MlflowTagValue]

    def __call__(self, value: T) -> MlflowTagValue:
        return self.serialize_fn(value)


@dataclass
class ParameterSerializer:
    # T isn't V because V is a value type, whereas T is a parameter
    # `serializers` is not a dict because the order matters.
    # e.g. a `datetime` is also a `date`. Then it has to prioritize the `datetime` serializer.
    serializers: list[_Serializer]

    def serialize(self, val: Any) -> MlflowTagValue:
        if val is None:
            return "null"
        val_type: type = type(val)
        for ser in self.serializers:
            if isinstance(val, ser.typ):
                return ser(val)
        raise TypeError(f"Got an unknown parameter type: {val_type}")


def identical_function(val: T) -> T:
    return val


default_serializer: ParameterSerializer = ParameterSerializer(
    [
        _Serializer[str](str, identical_function),
        _Serializer[int](int, identical_function),
        _Serializer[float](float, identical_function),
        _Serializer[bool](bool, lambda b: int(b)),
        _Serializer[datetime.date](
            datetime.date, lambda d: cast(datetime.date, d).isoformat()
        ),
    ]
)


def get_optional_serializer(
    serializer: Callable[[str], T]
) -> Callable[[str], Optional[T]]:
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


def deserialize_params(
    params: dict[str, Any],
    param_types: dict[str, type[Parameter]],
) -> dict[str, Any]:
    try:
        deserializers = {
            key: DESERIALIZERS[param_types[key].__name__] for key in params.keys()
        }
    except KeyError as e:
        raise UnknownParameter(str(e))
    return {key: deserializers[key](str(val)) for key, val in params.items()}
