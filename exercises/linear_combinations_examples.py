import numpy as np
import guide_to_linear_algebra.laguide as lag

# Example 1
A = np.array([
    [2, 2, -1],
    [-1, 2, -1],
    [0, 6, -1],
    [3, -4, 0]])
B = np.array([[4], [-2], [4], [2]])

AB = np.hstack((A, B))
print("AB =", AB)
R = lag.FullRowReduction(AB)
print("Reduced form AB:", "\n", R)
