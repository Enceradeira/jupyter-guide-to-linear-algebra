import numpy as np
from guide_to_linear_algebra.inner_products import unit_vector, magnitude, angle


def test_unit_vector_has_magnitude_one():
    V = np.array([[3], [-1], [2]])
    result = unit_vector(V)
    assert np.allclose(magnitude(result), 1.0)


def test_unit_vector_preserves_direction():
    V = np.array([[4], [0], [0]])
    result = unit_vector(V)
    assert np.allclose(result, np.array([[1], [0], [0]]))


def test_angle_orthogonal_vectors():
    U = np.array([[1], [-1], [2]])
    W = np.array([[2], [0], [-1]])
    assert np.allclose(angle(U, W), 0.0)


def test_angle_parallel_vectors():
    U = np.array([[1], [0], [0]])
    V = np.array([[3], [0], [0]])
    assert np.allclose(angle(U, V), 1.0)


def test_angle_opposite_vectors():
    U = np.array([[1], [0], [0]])
    V = np.array([[-3], [0], [0]])
    assert np.allclose(angle(U, V), -1.0)
