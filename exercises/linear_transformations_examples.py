import numpy as np
import guide_to_linear_algebra.solve_system as ss

# Example of a linear transformation T: R^2 -> R^3

# Linearity


def T(V):
    W = np.zeros((3, 1))
    W[0, 0] = 2*V[0, 0]
    W[2, 0] = V[1, 0]
    return W


V = np.array([[3], [5]])
W = T(V)

print(V, '\n\n', W)


# proof that T is linear by example
V1 = np.array([[1], [-7]])
V2 = np.array([[-2], [4]])
k = 2

print("k*T(V1) = T(k*V1):\n", k*T(V1), " =\n", T(k*V1))
print("T(V1 + V2) = T(V1) + T(V2):\n", T(V1 + V2), " =\n", T(V1) + T(V2))

# Example of matrix multiplication as a linear transformation


def L(V):
    A = np.array([[1, 1, 0], [1, 0, 2], [3, 1, -1]])
    W = A@V
    return W


V1 = np.array([[1], [3], [5]])
V2 = np.array([[0], [7], [-5]])
k = -3

print("k*L(V1) = L(k*V1):\n", k*L(V1), " =\n", L(k*V1))
print("L(V1 + V2) = L(V1) + L(V2):\n", L(V1 + V2), " =\n", L(V1) + L(V2))


# Composition
V = np.array([[3], [5]])
composition_output = L(T(V))
print(composition_output)

# Ivertability


def L_inverse(V):
    A = np.array([[1, 1, 0], [1, 0, 2], [3, 1, -1]])
    A_inverse = ss.Inverse(A)
    W = A_inverse@V
    return W


V = np.array([[1], [3], [5]])
print("L_inverse(L(V)) = V:\n", L_inverse(L(V)), " =\n", V)
