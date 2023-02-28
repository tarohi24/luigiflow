import datetime
from typing import Any

import pytest

from luigiflow.domain.serializer import default_serializer


class TestDefaultSerializer:
    @pytest.mark.parametrize(
        "val, expected",
        [
            (
                "hi",
                "hi",
            ),
            (
                100,
                100,
            ),
            (
                1.000,
                1.0,
            ),
            (
                True,
                1,
            ),
            (
                datetime.date(2021, 1, 2),
                "2021-01-02",
            ),
        ],
    )
    def test_serialize(self, val: Any, expected: Any):
        assert default_serializer.serialize(val) == expected
