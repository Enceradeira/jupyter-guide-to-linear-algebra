import numpy as np
from . import laguide as lag


def NrFreeVariables(E):
    R = lag.FullRowReduction(E)
    nr_bound_vars = 0
    nr_columns = R.shape[1]
    for i, row in enumerate(R):
        if np.all(row[:-1] == 0) and row[-1] != 0:
            return 0  # inconsistent system
        if row[i] == 1 and np.sum(row[:i] == 0):
            nr_bound_vars += 1

    return nr_columns - 1 - nr_bound_vars
