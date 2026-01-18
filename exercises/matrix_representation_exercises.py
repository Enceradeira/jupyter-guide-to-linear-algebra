import numpy as np
import guide_to_linear_algebra.laguide as lag
import guide_to_linear_algebra.solve_system as ss

print("Exercise 1a")


def B(V):
    x1 = V[0, 0]
    x2 = V[1, 0]
    x3 = V[2, 0]
    x4 = V[3, 0]
    return np.array([
        [x1 + 2*x2 - x3 - x4],
        [x2 - 3*x3 + 2 * x4]])


E = np.eye(4)

Br = np.hstack([B(E[:, [0]]), B(E[:, [1]]), B(E[:, [2]]), B(E[:, [3]])])
print("Matrix representation of B:\n", Br)

R = lag.FullRowReduction(Br)
print("Row reduced form of matrix representation of B:\n", R)
print("Matrix representation is onto")

print("\nExercise 1b")


def C(V):
    x1 = V[0, 0]
    x2 = V[1, 0]
    x3 = V[2, 0]
    return np.array([
        [x1-x2-8*x3],
        [4*x1 + 5*x2 - x3],
        [-x1 - x2 + 3 * x3]])


E = np.eye(3)
Cr = np.hstack([C(E[:, [0]]), C(E[:, [1]]), C(E[:, [2]])])

print("Matrix representation of C:\n", Cr)

R = lag.FullRowReduction(Cr)
print("Row reduced form of matrix representation of C:\n", R)
print("Matrix representation is onto, one-to-one and therefore invertible")


print("\nExercise 2")


A = np.array([
    [1, 1, 1],
    [2, 3, 4]])


def L(V):
    return A @ V


E = np.eye(3)


M = np.hstack([L(E[:, [0]]), L(E[:, [1]]), L(E[:, [2]])])
print("Matrix representation of L:\n", M)

R = lag.FullRowReduction(M)
print("Row reduced form of matrix representation of L:\n", R)
print("Matrix representation is onto but not one-to-one, therefore not invertible")

print("\nExercise 3")

M = np.array([
    [2, 1, 1],
    [0, 3, 2]])

X = np.array([
    [4],
    [5],
    [3]])

Lx = M @ X
print("L(X) =\n", Lx)

print("\nExercise 4")

print("[S] is onto, but not one-to-one, therefore not invertible")

print("\nExercise 5")

Wr = np.array([
    [1, 1, 0],
    [1, 2, 2],
    [2, 1, 3]])

Wx = np.array([
    [3],
    [11],
    [13]])

Wri = ss.Inverse(Wr)
X = Wri @ Wx
print("1. Solution X =\n", X)


X = ss.SolveSystem(Wr, Wx)
print("2. Solution X =\n", X)


print("\nExercise 6")

