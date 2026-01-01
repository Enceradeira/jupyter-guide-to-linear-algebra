import numpy as np
import guide_to_linear_algebra.laguide as lag
import guide_to_linear_algebra.plt_helper as ph


# Example 1: Polynomial Vector Space
A = np.array([
    [3, 0, 6, -2],
    [2, 1, 3, 2],
    [1, -5, 7, 8],
    [-1, 4, -6, 0]])


R = lag.FullRowReduction(A)
print("Reduced form of A:", "\n", R, "\n")

# Example 2: Matrices
