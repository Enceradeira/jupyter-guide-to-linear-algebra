import numpy as np
import guide_to_linear_algebra.laguide as lag

B = np.array([
    [2, 1],
    [3, 8]])

R = lag.FullRowReduction(B)
print("Row reduced form of B:\n", R)


def T(V):
    return np.array([
        2*V[0] + V[1],
        V[1]
    ])


V1 = T(B[:, 0])
V2 = T(B[:, 1])

print("T(V1):\n", V1)
print("T(V2):\n", V2)

X = np.array([1, 1])

A = np.column_stack([B, X])
R = lag.FullRowReduction(A)
print("Row reduced form of augmented matrix [T(B) | X]:\n", R)

C = R[:, 2]
print("Coordinates of X in the image of T:\n", C)

M = np.column_stack([V1, V2])
print("Matrix representation of T:\n", M)
Tx = M @ C
 
print("T(X):\n", Tx)
