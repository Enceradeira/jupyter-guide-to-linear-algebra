import numpy as np
import guide_to_linear_algebra.laguide as lag
import guide_to_linear_algebra.solve_system as ss
import scipy.linalg as sla

# Solve CY = I (whereas, by definition, Y is the inverse of C)
C = np.array([
    [1, 0, 2, 0],
    [3, 1, -3, 2],
    [2, 0, 4, 4],
    [2, 1, -1, 1]
])

I = np.eye(4)

I1 = I[:, 0:1]
I2 = I[:, 1:2]
I3 = I[:, 2:3]
I4 = I[:, 3:4]


CI1 = ss.SolveSystem(C, I1)
CI2 = ss.SolveSystem(C, I2)
CI3 = ss.SolveSystem(C, I3)
CI4 = ss.SolveSystem(C, I4)

C_inverse = np.hstack((CI1, CI2, CI3, CI4))
print(C_inverse)

# Solve CY = I
Y = I@C_inverse
print("Y (Inverse)\n", Y)

# Check by inserting I = CY
print("I (back substitution):\n", C @ Y)

# Use the generic method
print("Y (Inverse using generic method):\n", ss.Inverse(C))

print("# Exercise 1:")
# AX = B
# A Ai X = B Ai
# X = B Ai
A = np.array([
    [2, 3, 1],
    [3, 3, 1],
    [2, 4, 1]])
B = np.array([
    [4],
    [8],
    [5]])

Ai = sla.inv(A)
X = Ai @ B

print("Test Solution A@X=B:\n", A@X)

print("# Exercise 2:")


A = np.random.randint(low=0, high=100, size=(4, 4))
B = np.random.randint(low=0, high=100, size=(4, 4))

L = sla.inv(A@B)
R = sla.inv(B) @ sla.inv(A)
print("L - R = 0", np.round(L - R))

print("# Exercise 4:")
A = np.array([
    [1, 2, -3],
    [-1, 1, -1],
    [0, -2, 3]])
B = np.array([
    [1],
    [1],
    [1]])
Ai = sla.inv(A)
X = Ai @ B
print("Test Solution A@X=B:\n", A @ X)

print("# Exercise 5:")
A = np.array([
    [3, 1, 0],
    [5, 2, 1],
    [0, 2, 3]])
C = np.array([
    [1, 2, 1],
    [3, 4, 0],
    [1, 0, 2]])

AC = np.hstack((A, C))
ACR = ss.RowReduction(AC)  # Reduced matrix
ACRB = ACR[:, 0:3]
Y0 = ss.BackSubstitution(ACRB, ACR[:, 3:4])
Y1 = ss.BackSubstitution(ACRB, ACR[:, 4:5])
Y2 = ss.BackSubstitution(ACRB, ACR[:, 5:6])
Y = np.hstack((Y0, Y1, Y2))
print("Proof AY-C=0:\n", np.round(A @ Y - C))

print("# Exercise 6:")
A = np.random.randint(low=0, high=100, size=(4, 1))
B = np.random.randint(low=0, high=100, size=(1, 4))
# ABi = sla.inv(A @ B) -> raises an error
print("det(A)\n", np.linalg.det(A @ B))
print("A @ B:\n", A @ B)

print("# Exercise 7:")
A = np.random.randint(low=0, high=100, size=(3, 3))

Ati = sla.inv(A.T)
Ait = sla.inv(A).T
print("Proof Ati - Ait = 0:\n", np.round(Ati - Ait))

print("# Exercise 8:")
# det(A) != 0, then A is invertible. since all diagonal products are zero
# except one the condition can be reduced to: x2 != 0
x1 = 4
x2 = 1
x3 = 1
x4 = 3
A = np.array([
    [4, x1, 0, 0],
    [0, x2, 0, 0],
    [0, x3, 1, 0],
    [0, x4, 0, 3]])

print("A A^-1 = I:\n", np.round(A @ sla.inv(A)))

print("# Exercise 9:")
# AF = I
A = np.array([
    [1, 0, 0, 2],
    [0, -1, 1, 4]])
I = np.eye(2)

# there are many solutions for F, one intuitively found is:
F = np.array([[1, 0],
              [0, 0],
              [0, 1],
              [0, 0]])
print("AF = I:\n", np.round(A @ F))
