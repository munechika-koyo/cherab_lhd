import numpy as np
import pytest

from cherab.lhd.emc3.cython.utility import compute_centers


@pytest.mark.parametrize(
    "vertices, cells, indices, expected_centers",
    [
        (
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ]
            ),
            np.array([[0, 1, 2, 3, 4, 5, 6, 7]]),
            np.array([[[0]]], dtype=np.uint32),
            np.array([[[[0.5, 0.5, 0.5]]]]),
        ),
        (
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 0.0, 1.0],
                    [2.0, 1.0, 0.0],
                    [2.0, 1.0, 1.0],
                ]
            ),
            np.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, 8, 9, 2, 5, 10, 11, 6]]),
            np.array([[[0, 1]]], dtype=np.uint32),
            np.array([[[[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]]]]),
        ),
    ],
)
def test_compute_centers(vertices, cells, indices, expected_centers):
    centers = compute_centers(vertices, cells, indices)
    np.testing.assert_allclose(centers, expected_centers)
