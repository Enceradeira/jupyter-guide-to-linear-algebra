import numpy as np
from . import laguide as lag


def find_pivots(R):
    """
     Find the pivot columns in the row echelon form R of a matrix.
    """
    pivots = []

    for row in R:  # Exclude the last column (augmented part)
        if np.all(row == 0):
            continue  # Skip zero rows

        idx_none_zeros = np.nonzero(row != 0)[0]
        idx_one = np.nonzero(row == 1)[0][0]

        if np.all(idx_none_zeros >= idx_one):
            # a 1 only preceded by zeros
            pivots.append(idx_one)

    return np.array(pivots)


def nr_free_variables(E):
    # Find the number of columns in R that do not contain a pivot
    R = lag.FullRowReduction(E)
    P = find_pivots(R)

    return R.shape[1] - len(P) - 1


def has_no_solution(E):
    R = lag.FullRowReduction(E)

    # Return true if the last column is a pivot column in R
    P = find_pivots(R)

    idx_augmented_column = R.shape[1] - 1
    if np.any(P == idx_augmented_column):
        return 1
    else:
        return 0
