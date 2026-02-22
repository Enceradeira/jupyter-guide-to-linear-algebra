import numpy as np
from guide_to_linear_algebra.web_navigation import web_navigation_matrix


def test_web_navigation_matrix():
    # Page 1 -> 2, Page 2 -> 1,3, Page 3 -> 1
    adjacency = np.array([[0, 1, 1],
                          [1, 0, 0],
                          [0, 1, 0]])

    expected = np.array([[0,   0.5, 0.9],
                         [0.9, 0,   0.1],
                         [0.1, 0.5, 0]])

    result = web_navigation_matrix(adjacency)
    np.testing.assert_array_almost_equal(result, expected)


def test_web_navigation_matrix_no_links():
    # Page 1 -> 2, Page 2 -> no links, Page 3 -> 1
    adjacency = np.array([[0, 0, 1],
                          [1, 0, 0],
                          [0, 0, 0]])

    expected = np.array([[0,   0.5, 0.9],
                         [0.9, 0,   0.1],
                         [0.1, 0.5, 0]])

    result = web_navigation_matrix(adjacency)
    np.testing.assert_array_almost_equal(result, expected)
