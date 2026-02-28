import numpy as np
from guide_to_linear_algebra.laguide import DotProduct


def magnitude(V):
    return np.sqrt(DotProduct(V, V))


def unit_vector(V):
    return V / magnitude(V)


def angle(U, V):
    return DotProduct(U, V) / (magnitude(U) * magnitude(V))
    pass
