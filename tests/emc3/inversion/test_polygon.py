import numpy as np
import pytest

from cherab.lhd.emc3.inversion.polygon import test_polygon_area as polygon_area


@pytest.mark.parametrize(
    "vertices, expected_area",
    [
        (np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64), 1.0),  # Square
        (np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64), 0.5),  # Triangle
        (np.array([[0, 0], [2, 0], [1, 1], [0, 2]], dtype=np.float64), 2.0),  # Irregular polygon
    ],
)
def test_polygon_area(vertices, expected_area):
    assert polygon_area(vertices) == expected_area
