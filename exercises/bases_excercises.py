import numpy as np
import guide_to_linear_algebra.laguide as lag

# Exercise 1
# Geg:
P = np.array([
    [1, 0, 3, -2, 4],
    [-1, 1, 6, -2, 1],
    [-2, 1, 3, 0, -3]])

# Ges: Basis for the set of solutions of P X = 0
R = lag.FullRowReduction(P)
print("Reduced form P:", "\n", R)

# There are 3 free variables, x3 = r, x4 = s, x5 = t
# x1 = -3r + 2s -4t
# x2 = -9r + 4s -5t
# x3 = r
# x4 = s
# x5 = t

# Therefore
B = np.array([
    [-3, 2, -4],
    [-9, 4, -5],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
print("Basis B for the set of solutions of P X = 0:", "\n", B, "\n")

rst = np.array([2, 3, 4])
X = B @ rst
print("X for r,s,t :", "\n", X, "\n")
print("P X:", "\n", P @ X, "\n")  # Should be zero vector
