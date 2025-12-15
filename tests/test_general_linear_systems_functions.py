import numpy as np
from guide_to_linear_algebra.general_linear_systems_functions import nr_free_variables, find_pivots, has_no_solution


def test_find_pivots_example_1():
    E = np.array([[1, 0, 0, 5],
                  [0, 1, 0, -3],
                  [0, 0, 1, 2]])

    assert np.array_equal(find_pivots(E), [0, 1, 2])


def test_find_pivots_example_2():
    E = np.array([[1, 0, 0, 5],
                  [0, 1, 0, -3],
                  [0, 0, 0, 0]])

    assert np.array_equal(find_pivots(E), [0, 1])


def test_find_pivots_example_3():
    E = np.array([[1, 2, 0, 4],
                  [0, 0, 1, -1],
                  [0, 0, 0, 0]])
    assert np.array_equal(find_pivots(E), [0, 2])


def test_find_pivots_example_4():
    E = np.array([[1, 3, -2, 7],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    assert np.array_equal(find_pivots(E), [0])


def test_find_pivots_example_5():
    E = np.array([[0, 1, 2, 3],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    assert np.array_equal(find_pivots(E), [1])


def test_nr_free_variables_unique_solution():
    E = np.array([[1, 0, 0, 5],
                  [0, 1, 0, -3],
                  [0, 0, 1, 2]])
    assert nr_free_variables(E) == 0


def test_nr_free_variables_one_free_variable():
    E = np.array([[-2, 2, -2, 2, 0],
                  [1, -2, -2, 0, -1],
                  [1, 0, 2, -2, 1]])
    assert nr_free_variables(E) == 1


def test_nr_free_variables_two_free_variables():
    E = np.array([[-2, 4, 2, -8, 4, -8],
                  [3, -6, -2, 11, -7, 13],
                  [1, - 2, -5, 8, 1, -3]])
    assert nr_free_variables(E) == 2


def test_has_no_solution_example_1():
    E = np.array([
        [1, -2, 2, 0],
        [2, 2, 2, 1],
        [0, -1, -1, -2],
        [-2, -1, -1, 0]])

    assert has_no_solution(E) == 1


def test_has_no_solution_example_2():
    E = np.array([[-2, 2, -2, 2, 0],
                  [1, -2, -2, 0, -1],
                  [1, 0, 2, -2, 1]])

    assert has_no_solution(E) == 0


def test_has_no_solution_example_3():
    E = np.array([[1, -2, 2, 0],
                  [0, -1, -1, -2],
                  [-2, -1, -1, 0]])

    assert has_no_solution(E) == 0
