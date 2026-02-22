import numpy as np


def web_navigation_matrix(A):
    rows, cols = A.shape
    r = 0.2 / (cols - 1)
    k = A.sum(axis=0)

    W = A * (0.8 / k) + r

    # special case for pages with no outgoing links
    no_links = k == 0
    k[no_links] = 1
    W[:, no_links] = 1 / (cols - 1)

    np.fill_diagonal(W, 0)
    return W
