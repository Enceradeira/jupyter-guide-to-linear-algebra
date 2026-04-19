import numpy as np
import guide_to_linear_algebra.laguide as lag

# Exercise 1: Determine whether the columns of U are linearly independent.
U = np.array([
    [0, 1, 3],
    [5, -1, 2],
    [2, 0, 2],
    [2, -1, -1]])

R = lag.FullRowReduction(U)
print("Reduced form of U:", "\n", R)

# The last vector is linear depndend. U3 = U1 + 3*U2
print("U3 = U1 + 3U2: ", np.array_equal(U[:, 0:1] + 3*U[:, 1:2], U[:, 2:3]))

# Exercise 2: Are the four vectors below linearly independent?
w1 = np.array([1, 0, 0, 1])
w2 = np.array([0, 1, -1, 0])
w3 = np.array([-1, 0, -1, 0])
w4 = np.array([1, 1, 1, -1])

W = np.column_stack((w1, w2, w3, w4, np.array([[0], [0], [0], [0]])))
R = lag.FullRowReduction(W)
print("Reduced form of W:", "\n", R)

print("U4 = -1*U1 + 1*U2 -2 *U3: ", np.array_equal(-1*w1 + 1*w2 - 2*w3, w4))

# Alternatively, without reading it out from R
# x1 * U1 + x2 * U2 + x3 * U3 = U4

T = np.column_stack((w1, w2, w3, w4))
T = lag.FullRowReduction(T)
print("Reduced form of T:", "\n", T)

# Exercise 3
b = 6  # manually found by gauss elimination

X = np.array([
    [1, 1, 2],
    [0, 2, 4],
    [1, 3, b]])

XR = lag.FullRowReduction(X)
print("Reduced form of X:", "\n", XR)

print("X3 = 0*X1 + 2*X2: ", np.array_equal(0 *
      X[:, 0:1] + 2*X[:, 1:2], X[:, 2:3]))

# Exercise 4

# Exercise 5: Show AX=B does not have infinitely many solutions.
A = np.array([
    [1, 1, 2],
    [2, 0, 1],
    [3, 1, 1]])

B = np.array([
    [1],
    [1],
    [1]])

AB = np.hstack((A, B))
R = lag.FullRowReduction(AB)
print("Reduced form of AB:", "\n", R)

# there is a pivot in each row -> no free variables -> no infinitely many solutions

# Exercise 6: Find a nonzero vector in the null space of A.
A = np.array([
    [1, 2, 3],
    [1, 0, 1],
    [1, 1, 2]])

R = lag.FullRowReduction(A)
print("Reduced form of A:", "\n", R)

a = 7
N = np.array([[-a], [-a], [a]])
print("A N = 0: ", np.array_equal(A @ N, np.zeros((3, 1))))

# Exercise 7: Find two distinct nonzero vectors in the null space of D.
D = np.array([
    [4, 4, 3],
    [8, 8, 6],
    [1, 0, 1]])

R = lag.FullRowReduction(D)
print("Reduced form of D:", "\n", R)

a = 3
N = np.array([[-4*a], [1*a], [4*a]])
print("D N = 0: ", np.array_equal(D @ N, np.zeros((3, 1))))

# Excercise 8

A = np.array([
    [1, 2, 0],
    [0, 1, 1]])
Xh = np.array([[-3], [2], [0]])

AXh = A @ Xh
print("AXh: ", AXh)

AXp = np.array([[0], [0]])
B = AXh + AXp
print("B: ", B)

t = 9
X = np.array([
    [-3 + 2*t],
    [2 - t],
    [t]])

print("X: ", X)
print("AX = B: ", np.array_equal(A @ X, B))
