from guide_to_linear_algebra import solve_system
import numpy as np


def run_RREF(A, E):
    """
    Run the RREF function on matrix A and check if it equals E.
    """
    S = solve_system.RREF(A)
    try:
        assert np.allclose(S, E)
    except AssertionError:
        print("A matrix:\n", A)
        print("S matrix:\n", S)
        print("E matrix:\n", E)
        raise


def test_RREF_with_example_1():
    A = np.array(
        [[3,  2,  1,  2],
         [-7, 1,  0,  2],
         [3,  2,  2,  7]])

    R = np.array(
        [[1, 0, 0, -0.41176471],
         [0, 1, 0, -0.88235294],
         [0, 0, 1, 5.]]
    )

    run_RREF(A, R)


def test_RREF_with_example_2():
    A = np.array(
        [[4, -8, 5, -14,],
         [-2, -1, 0, -23],
         [3,  8, 5, 123]])

    R = np.array(
        [[1, 0, 0, 7],
         [0, 1, 0, 9],
         [0, 0, 1, 6]]
    )

    run_RREF(A, R)
