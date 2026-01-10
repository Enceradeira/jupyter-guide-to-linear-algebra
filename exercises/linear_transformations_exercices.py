import numpy as np
import guide_to_linear_algebra.laguide as lag

# Exercise 1
print("Exercise 1")


def T(V):
    W = np.zeros((3, 1))
    W[0, 0] = 2*V[0, 0]
    W[2, 0] = V[1, 0]
    return W


V = np.array([[1], [3]])

print("T(V):\n", T(V))

# Exercise 2
print("Exercise 2")


def T_inv(V):
    W = np.array([
        [V[0, 0] / 2],
        [V[2, 0]]
    ])
    return W


TU = np.array([[5], [0], [-1]])

U = T_inv(TU)
print("U:\n", U)
print("T(U):\n", T(U))


# Exercise 3
print("Exercise 3")


def N(V):
    return np.array([
        [8*V[1, 0]],
        [V[0, 0] + V[1, 0] + 3]])


# Test linearity of N
k = 4
U = np.array([[3], [-3], [3]])
V = np.array([[-5], [0], [1]])

print("k*N(U) = N(k*U):\n", k*N(U), " =\n", N(k*U))
print("N(U + V) = N(U) + N(V):\n", N(U + V), " =\n", N(U) + N(V))
print("N is not liear, above equalities do not hold.")


# Exercise 4
print("Exercise 4")


def S(V):
    return np.array([
        [V[0, 0] + 2*V[1, 0]],
        [3*V[2, 0]]])


def R(V):
    M = np.array([
        [3, 0],
        [-1, 1]])
    return M@V


def RS(V):
    return np.array([
        [3, 0],
        [-1, 1]]) @ np.array([
            [V[0, 0] + 2*V[1, 0]],
            [3*V[2, 0]]])


T = np.array([[3], [-2], [1]])
print("compostion R(S(V)):\n ", R(S(T)))
print("compsotion RS(V):\n ", RS(T))

# SR is not defined since the output of R is 2D and the input of S is 3D

# Exercise 5
print("Exercise 5")


def S(V):
    return np.array([
        [V[0, 0] + V[1, 0]],
        [1],
        [V[2, 0] + V[0, 0]]])

# investigate linearity of S


k = 2
U = np.array([[1], [0], [3]])
V = np.array([[4], [-1], [2]])

print("k*S(U) = S(k*U):\n", k*S(U), " =\n", S(k*U))
print("S(U + V) = S(U) + S(V):\n", S(U + V), " =\n", S(U) + S(V))
print("S is not linear, above equalities do not hold.")


def T(V):
    return np.array([
        [V[0, 0] + V[1, 0]],
        [0],
        [V[2, 0] + V[0, 0]]])


# investigate linearity of T
print("k*T(U) = T(k*U):\n", k*T(U), " =\n", T(k*U))
print("T(U + V) = T(U) + T(V):\n", T(U + V), " =\n", T(U) + T(V))
print("T is linear, above equalities hold.")

# Exercise 6
print("Exercise 6")

# L(kU+V) = L(kU) + L(V) = kL(U) + L(V) = 7 [1 1] = [3 1] = [10 8]
k = 7
LU = np.array([[1], [1]])
LV = np.array([[3], [1]])
result = k*LU + LV
print("L(kU + V):\n", result)


# Exercise 7
print("Exercise 7")

A = np.array([
    [1, 0, 2],
    [2, 1, 1]])
Y = np.array([[1], [2]])
R = lag.FullRowReduction(np.hstack([A, Y]))
print("Reduced form A:", "\n", R)

a = 1
x1 = 1-2*a
x2 = 3*a
x3 = a

X = np.array([[x1], [x2], [x3]])

print("Test A X = Y:\n", A @ X)


# Exercise 8
print("Exercise 8")

