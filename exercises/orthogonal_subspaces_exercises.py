import numpy as np
import guide_to_linear_algebra.laguide as lag

# Exercise 1
# Let B be the following 4x3 matrix.
# Find bases for C(B^T) and N(B), the row space and null space of B.
print("Exercise 1")

B = np.array(
    [[4, -3, -2],
     [2, 0, 2],
     [0, 1, 2],
     [1, 1, 3]]
)

zero = np.zeros((4, 1))
B_augmented = np.hstack((B, zero))

# Null space basis (dimension 1)
B_reduced = lag.FullRowReduction(B_augmented)
print("B_reduced:\n", B_reduced, "\n")

t = 8
X0 = np.array([[-t], [-2*t], [t]])
print("B @ X0:\n", B @ X0)  # should be all zeros

C_Bt_reduced = lag.FullRowReduction(B.T)
print("C(B^T) reduced:\n", C_Bt_reduced, "\n")

CBt = np.array([
    [4, 2],
    [-3, 0],
    [-2, 2]])
print("C(B^T):\n", CBt)

NB = np.array([[-1], [-2], [1]])
print("NB:\n", NB)

# Exercise 2
X = np.array([[8], [1], [-2]])

PE_system = np.hstack((NB, CBt, X))
print("PE_system:\n", PE_system)
PE_reduced = lag.FullRowReduction(PE_system)
print("PE_reduced:\n", PE_reduced)

PEX = np.array([[-3], [5/3], [-5/6]])
print("PE_System @ PEX:\n", PE_system[:, 0:3] @ PEX)

# Exercise 3
A = np.array([
    [2, 1, 2, 0],
    [3, 0, 1, 1],
    [1, 1, 1, 0]])

A_reduced = lag.FullRowReduction(A)
print("A_reduced:\n", A_reduced)

AT_reduced = lag.FullRowReduction(A.T)
print("A^T reduced:\n", AT_reduced)

# Exercise 4
U1 = np.array([[1], [0], [2], [2]])
U2 = np.array([[-2], [1], [0], [-1]])
U = np.hstack((U1, U2))
UT_reduced = lag.FullRowReduction(U.T)
print("UT_reduced:\n", UT_reduced)


def calc_U(s, t):
    return np.array([[-2*s - 2*t],
                     [-4*s - 3*t],
                     [s],
                     [t]])


U3 = calc_U(1, 0)  # s = 1, t = 0
U4 = calc_U(0, 1)  # s = 0, t = 1

print("U3:\n", U3)
print("U4:\n", U4)

print(U.T @ np.hstack((U3, U4)))  # should be all zeros
print("U.T @ {U3, U4} = 0", np.array_equal(U.T @
      np.hstack((U3, U4)), np.zeros((2, 2))))

# Exercise 5
# W is a subspace of R^5 spanned by {W1, W2, W3}. Find a basis for W⊥.
W1 = np.array([[1], [1], [0], [1], [1]])
W2 = np.array([[3], [2], [0], [1], [1]])
W3 = np.array([[0], [1], [1], [1], [2]])
W = np.hstack((W1, W2, W3))
WT_reduced = lag.FullRowReduction(W.T)
print("WT_reduced:\n", WT_reduced)


def calc_W(s, t):
    return np.array([[s + t],
                     [-2*s - 2*t],
                     [s],
                     [s],
                     [t]])


W4 = calc_W(1, 0)  # s = 1, t = 0
W5 = calc_W(0, 1)  # s = 0, t = 1

print("W.T @ {W4, W5} = 0", np.array_equal(W.T @
      np.hstack((W4, W5)), np.zeros((3, 2))))


# Exercise 6
# U = span{V1, V2}, W = span{V3, V4}. Are U and W orthogonal complements?
V1 = np.array([[1], [0], [2], [1]])
V2 = np.array([[1], [-1], [2], [0]])
V3 = np.array([[-1], [1], [1], [-1]])
V4 = np.array([[-2], [0], [1], [0]])
V = np.hstack((V1, V2, V3, V4))
V_reduced = lag.FullRowReduction(V)
print("V_reduced:\n", V_reduced)

U = np.hstack((V1, V2))
W = np.hstack((V3, V4))
print("U.T @ W = 0", np.array_equal(U.T @ W, np.zeros((2, 2))))

# Exercise 7
# Find P in C(A) and E orthogonal to P such that B = P + E.
A = np.array([
    [1, 2],
    [2, 1],
    [2, -2]])
B = np.array([[1], [1], [1]])
AT_reduced = lag.FullRowReduction(A.T)
print("AT_reduced:\n", AT_reduced)

NAT = np.array([[2], [-2], [1]])

CA = A
print("CA.T @ NAT", np.array_equal(CA.T @ NAT, np.zeros((2, 1))))

PE_system = np.hstack((CA, NAT, B))
print("PE_system:\n", PE_system)

PE_reduced = lag.FullRowReduction(PE_system)
print("PE_reduced:\n", PE_reduced)

P = np.array([[7/9], [11/9], [8/9]])
E = np.array([[2/9], [-2/9], [1/9]])

print("P + E = B", np.array_equal(P + E, B))
print("P . E = 0", np.isclose(lag.DotProduct(P, E), 0))

# Exercise 8
S = np.array([
    [2, 3, 1, 0],
    [3, -3, 0, -6]])

S_reduced = lag.FullRowReduction(S)
print("S_reduced:\n", S_reduced)

a = -1
b = 1

a = -6
b = -4

U = np.array([
    [b, 2],
    [a, -3],
    [2, 3]])

W = np.array([
    [3],
    [b],
    [a]])

print("U.T @ W:\n", U.T @ W)
print("U.T @ W = 0", np.array_equal(U.T @ W, np.zeros((2, 1))))

# Exercise 10

V = np.array([[-1], [0], [8]])

U = np.array([
    [1, 2],
    [0, 2],
    [1, 3]])

UT_reduced = lag.FullRowReduction(U.T)
print("U^T reduced:\n", UT_reduced)

# chosen to be orthogonal to U by setting x3 = 2
W = np.array([[-2], [-1], [2]])
print("U.T @ W = 0", np.array_equal(U.T @ W, np.zeros((2, 1))))

S = np.hstack((U, W, V))
S_reduced = lag.FullRowReduction(S)
print("S_reduced:\n", S_reduced)

# Exercise 11
# (a) Find bases for the four fundamental subspaces of A:
#     C(A), N(A), R(A) = C(A^T), and N(A^T).
# (b) Show each row space basis vector is orthogonal to each null space basis vector.
# (c) Show each column space basis vector is orthogonal to each left null space basis vector.
print("Exercise 11")

A = np.array([
    [1, 1, 0, 2, -1],
    [1, 0, 1, -1, 0],
    [3, 1, 2, 0, -1]])
print("A:\n", A)

A_reduced = lag.FullRowReduction(A)
print("A_reduced:\n", A_reduced)

NA = np.array([
    [-1, 1, 0],
    [1, -3, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

print("A_reduced @ NA:\n", A_reduced @ NA)  # should be all zeros

AT_reduced = lag.FullRowReduction(A.T)
print("A^T reduced:\n", AT_reduced)

NAT = np.array([-1, -2, 1])

CA = A[:, 0:2]
print("CA:\n", CA)

RA = A.T[:, 0:2]
print("RA:\n", RA)

# Exercise 11 (b)
print("RA.T @ NA:\n", RA.T @ NA)
print("RA.T @ NA = 0:", np.array_equal(RA.T @ NA, np.zeros((2, 3))))

# Exercise 11 (c)
print("CA.T @ NAT:\n", CA.T @ NAT)
print("CA.T @ NAT = 0:", np.array_equal(CA.T @ NAT, np.zeros(2)))

# Exercise 12a
A = np.array([[1, 2, -3]])
M = np.array([
    [-2, 3],
    [1, 0],
    [0, 1]])

print("A @ M:\n", A @ M)
print("A @ M = 0:", np.array_equal(A @ M, np.zeros((1, 2))))


A2 = np.array([[1, -1/3, -2/3]])
M2 = np.array([
    [1, 2],
    [3, 0],
    [0, 3]])

print("A2 @ M2:\n", A2 @ M2)
print("A2 @ M2 = 0:", np.array_equal(A2 @ M2, np.zeros((1, 2))))

# Exercise 121

E = np.array([
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, -3]])

E_reduced = lag.FullRowReduction(E)
print("E_reduced:\n", E_reduced)
