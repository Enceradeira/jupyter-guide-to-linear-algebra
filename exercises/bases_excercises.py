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

# Exercise 2
# Determine if the following set of vectors is a basis for R^4
W = np.array([
    [-1, 2, 0, -1],
    [0, 1, 0, 0],
    [1, 2, 1, -1],
    [2, 4, 0, 1]])

R = lag.FullRowReduction(W)
print("Reduced form W:", "\n", R)

# W has 4 pivot columns, therefore it describes a linearly independent set
# that spans R^4

# Exercise 3
# Give an example of a set of three vectors that does NOT form a basis for R^3.
# Provide a calculation that shows why the example is not a basis.

V = np.array([
    [-6, -2, 0],
    [3, 1, 6],
    [3, 1, 0]])

R = lag.FullRowReduction(V)
print("Reduced form V:", "\n", R)
# V has only 2 pivot columns, therefore it describes a linearly dependent set
# and only span R^2

# Exercise 4
# Calculate the dimension of the span of {U1, U2, U3, U4}
# The dimension of a vector space (or subspace) is defined as the number of
# vectors in any basis for the space.
U = np.array([
    [1, 2, 3, 5],
    [2, -3, -1, -4],
    [-1, 3, 2, 4],
    [3, -2, 1, -1]])
R = lag.FullRowReduction(U)
print("Reduced form U:", "\n", R)

# There is a pivot in row 1,2 and 4, therefore the basis spans the subspace of R^3
# and is B = {U1, U2, U4}
B = U[:, [0, 1, 3]]
print("Basis B:", "\n", B)

R_B = lag.FullRowReduction(B)
print("RREF of B:", "\n", R_B)

# Exercise 5
V = np.array([
    [1, 1, 1],
    [2, 0, 3],
    [1, 2, 1],
    [1, 2, 2]])
print("V:", "\n", V)
R_V = lag.FullRowReduction(V)
print("RREF of V:", "\n", R_V)

# Invent a fourth vector and check that it is still linearly independent
V4 = np.array([[1], [1], [1], [1]])
V_V4 = np.hstack((V, V4))
R_V_V4 = lag.FullRowReduction(V_V4)
print("RREF of (V|V4):", "\n", R_V_V4)

# Exercise 6
V = np.array([
    [1, 2],
    [2, 3],
    [0, 0],
    [0, 0]])
print("V:", "\n", V)
R_V = lag.FullRowReduction(V)
print("RREF of V:", "\n", R_V)

# The dimension is 2, since there are 2 pivot columns

# Exercise 7
# Find the value(s) of a for which {X1, X2, X3} is NOT a basis for R^3
# With a = 1, there are only 2 pivot columns, therefore the set is linearly dependent and does not span R^3
a = 1
X = np.array([
    [1, 2, 1],
    [2, a, 2],
    [1, 3, a]])
print("X:", "\n", X)
R_X = lag.FullRowReduction(X)
print("RREF of X:", "\n", R_X)

# Exercise 8
a, b, c = 1, 2, 3  # given linear combiantion coefficients

# anyU is R^5 -> meaning 5 rows/coordinates -> subspace containing all possbile solutions/linear combinations
anyU = np.array([[a], [a], [0], [b], [c]])

B = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])  # basis can be multipied by any combination of a,b,c to get any U

X = np.array([[a], [b], [c]])  # a particular solution
U = B @ X
print("AnyU:", "\n", anyU)
print("B @ X = U:", "\n", U)

# Exercise 9
# Find the coordinates of V with respect to basis β = {U1, U2, U3}
B = np.array([
    [1, 2, 3],
    [2, 1, 2],
    [3, 0, 5]])
V = np.array([[8], [6], [8]])

print("B:", "\n", B)
print("V:", "\n", V)
R_BV = lag.FullRowReduction(np.hstack((B, V)))
print("RREF of (B|V):", "\n", R_BV)

coordinates = np.array([[1], [2], [1]])
print("Coordinates of V with respect to basis β:", "\n", coordinates)
print("B @ coordinates = V:", "\n", B @ coordinates)

# Exercise 10
B = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]])

# Is B a basis for R^3? No there will always be a pivot missing in the last column,
# therefore it is not linearly independent and does not span R^3
