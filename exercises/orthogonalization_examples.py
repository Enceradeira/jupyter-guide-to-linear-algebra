import numpy as np
import guide_to_linear_algebra.laguide as lag
from guide_to_linear_algebra.inner_products import magnitude, unit_vector, angle

# Example 1: Gram-Schmidt algorithm
# Convert {V1, V2, V3} into an orthonormal set {U1, U2, U3}

V1 = np.array([[1], [2], [0]])
V2 = np.array([[1], [1], [1]])
V3 = np.array([[3], [0], [1]])

# Step 1: Scale V1 to unit length
U1 = unit_vector(V1)
print("U1:", U1.T)

# Step 2: Remove V2's component along U1, then normalize
W2 = V2 - lag.DotProduct(V2, U1) * U1
U2 = unit_vector(W2)
print("U2:", U2.T)

# Step 3: Remove V3's components along U1 and U2, then normalize
W3 = V3 - lag.DotProduct(V3, U1) * U1 - lag.DotProduct(V3, U2) * U2
U3 = unit_vector(W3)
print("U3:", U3.T)

# Verify orthogonality
print("U1·U2 = 0:", np.allclose(lag.DotProduct(U1, U2), 0))
print("U1·U3 = 0:", np.allclose(lag.DotProduct(U1, U3), 0))
print("U2·U3 = 0:", np.allclose(lag.DotProduct(U2, U3), 0))

# Verify unit magnitude
print("||U1|| = 1:", np.allclose(magnitude(U1), 1))
print("||U2|| = 1:", np.allclose(magnitude(U2), 1))
print("||U3|| = 1:", np.allclose(magnitude(U3), 1))

# Example 2: Orthogonal matrices
# A matrix Q with orthonormal columns satisfies Q^T Q = I
# When Q is square, Q^T = Q^(-1)

Q = np.hstack([U1, U2, U3])
print("\nExample 2: Orthogonal matrix")
print("Q:\n", Q)
print("Q^T Q:\n", Q.T @ Q)
print("Q^T Q = I:", np.allclose(Q.T @ Q, np.eye(3)))
