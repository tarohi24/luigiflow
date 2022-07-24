# I implemented this module becasue `luigi` doesn't offer some optional params,
# and even though some params are implemented,
# their name may be confusing (e.g. The type of `OptionalParameter`is `Optional[int]`,
# while that of `Parameter` is `str`.
from luigi import Parameter


class OptionalIntParameter(Parameter):
    ...


class OptionalFloatParameter(Parameter):
    ...


class OptionalStrParameter(Parameter):
    ...


class OptionalDateParameter(Parameter):
    ...
