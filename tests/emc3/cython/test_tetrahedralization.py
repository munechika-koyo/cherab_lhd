import numpy as np
import pytest

from cherab.lhd.emc3.cython.tetrahedralization import tetrahedralize


@pytest.mark.parametrize(
    "cells, expected",
    [
        (
            np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.uint32),
            np.array(
                [
                    [6, 2, 1, 0],
                    [7, 3, 2, 0],
                    [0, 7, 6, 2],
                    [1, 5, 6, 4],
                    [0, 4, 6, 7],
                    [6, 4, 0, 1],
                ],
                dtype=np.uint32,
            ),
        ),
        (
            np.array([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=np.uint32),
            np.array(
                [
                    [14, 10, 9, 8],
                    [15, 11, 10, 8],
                    [8, 15, 14, 10],
                    [9, 13, 14, 12],
                    [8, 12, 14, 15],
                    [14, 12, 8, 9],
                ],
                dtype=np.uint32,
            ),
        ),
        (
            np.array([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]], dtype=np.uint32),
            np.array(
                [
                    [6, 2, 1, 0],
                    [7, 3, 2, 0],
                    [0, 7, 6, 2],
                    [1, 5, 6, 4],
                    [0, 4, 6, 7],
                    [6, 4, 0, 1],
                    [14, 10, 9, 8],
                    [15, 11, 10, 8],
                    [8, 15, 14, 10],
                    [9, 13, 14, 12],
                    [8, 12, 14, 15],
                    [14, 12, 8, 9],
                ],
                dtype=np.uint32,
            ),
        ),
    ],
)
def test_tetrahedralize(cells, expected):
    result = tetrahedralize(cells)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "cells, exception, message",
    [
        (
            np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint32),
            ValueError,
            "cells must be a 2 dimensional array.",
        ),
        (
            np.array([[0, 1, 2, 3, 4, 5, 6]], dtype=np.uint32),
            ValueError,
            "cells must have a shape of (N, 8).",
        ),
    ],
)
def test_tetrahedralize_exceptions(cells, exception, message):
    with pytest.raises(exception) as excinfo:
        tetrahedralize(cells)
    assert str(excinfo.value) == message
