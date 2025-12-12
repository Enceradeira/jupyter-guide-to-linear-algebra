import numpy as np
import guide_to_linear_algebra.laguide as lag
import guide_to_linear_algebra.solve_system as ss
import scipy.linalg as sla

print("# Example 1:")

A = np.array([
    [3, -1, -2],
    [6, -1, -0],
    [-3, 5, 20]])

L = np.array([
    [1, 0, 0],
    [2, 1, 0],
    [-1, 4, 1]])

U = np.array([
    [3, -1, -2],
    [0, 1, 4],
    [0, 0, 2]])

print("LU:\n", L @ U)

print("# Elementary matrices:")
I = np.eye(4)
E = lag.RowAdd(I, 1, 2, -3)

print("I:\n", I)
print("E:\n", E)
A = np.array([
    [1, 2, 0, -1],
    [-1, 1, -1, 4],
    [2, 13, -4, 9],
    [-2, 5, -3, 13]])
Ac = A.copy()
# the same operation as on the elementary matrix
Ace = lag.RowAdd(Ac, 1, 2, -3)
print("A:\n", A)
print("E@A:\n", E @ A)
print("Ace:\n", Ace)

print("# Example 2:")
A = np.array([
    [1, 2, 0, -1],
    [-1, 1, -1, 4],
    [2, 13, -4, 9],
    [-2, 5, -3, 13]])
# Following transforms A into an upper triangular matrix
A1 = lag.RowAdd(A, 0, 1, 1)
A2 = lag.RowAdd(A1, 3, 2, 1)
A3 = lag.RowAdd(A2, 1, 2, -6)
A4 = lag.RowAdd(A3, 0, 3, 2)
A5 = lag.RowAdd(A4, 1, 3, -3)
print("A5 (A transformed into upper transformed matrix):\n", A5)
# We can now apply the same operations to I constucting the elementary matrices
E1 = lag.RowAdd(I, 0, 1, 1)
E2 = lag.RowAdd(I, 3, 2, 1)
E3 = lag.RowAdd(I, 1, 2, -6)
E4 = lag.RowAdd(I, 0, 3, 2)
E5 = lag.RowAdd(I, 1, 3, -3)
# Multiplying the elementary matrices in the same order as the operations and A
# gives us the same upper triangular matrix
print("Elementary matrices  (with same transformations) applied to A:\n",
      E5@E4@E3@E2@E1@A)

print("E3:\n", E3)
E3_inv = sla.inv(E3)
print("E3^-1:\n", E3_inv)
print("E3 @ E3^-1:\n", E3 @ E3_inv)

L1 = np.array([[1, 0, 0, 0], [-1, 1, 0, 0], [2, 3, 1, 0], [-2, 3, 0, 1]])
L2 = np.array([[1, 0, 0, 0], [2, 1, 0, 0], [-5, 4, 1, 0], [4, 4, 1, 1]])
print(L1, '\n')
print(L2, '\n')
print(L1@L2)

print("E5:\n", E5)
print("E4:\n", E4)
print("E3:\n", E3)

print("E5^-1\n", sla.inv(E5))
print("E4^-1 @ E5^-1\n", sla.inv(E4)@sla.inv(E5))
print("E3^-1 @ E4^-1 @ E5^-1\n", sla.inv(E3)@sla.inv(E4)@sla.inv(E5))

print("# Permutation matrices:")
C = np.random.randint(-6, 6, size=(4, 4))
P = lag.RowSwap(I, 1, 2)

print("C:", '\n', C, '\n', sep='')
print("P:", '\n', P, '\n', sep='')
print("PC:", '\n', P @ C, '\n', sep='')

print("# Example 3:")
B = np.array([
    [1, 2, -1, -1],
    [4, 8, -4, 2],
    [1, 1, 1, 2],
    [3, 3, 4, 4]])
print(B)
B1 = lag.RowAdd(B, 0, 1, -4)
L1 = lag.RowAdd(I, 0, 1, -4)

B2 = lag.RowAdd(B1, 0, 2, -1)
L2 = lag.RowAdd(I, 0, 2, -1)

B3 = lag.RowAdd(B2, 0, 3, -3)
L3 = lag.RowAdd(I, 0, 3, -3)
print("B after row operations:\n", B)

B4 = lag.RowSwap(B3, 1, 2)
P1 = lag.RowSwap(I, 1, 2)
print("B after row swap:\n", B)

B5 = lag.RowAdd(B4, 1, 3, -3)
L4 = lag.RowAdd(I, 1, 3, -3)
print("B after 2. row operation:\n", B)

B6 = lag.RowSwap(B5, 2, 3)
P2 = lag.RowSwap(I, 2, 3)
print("B after 2. row swap:\n", B)

U = P2@L4@P1@L3@L2@L1@B
print("U:\n", U)

possible_L = sla.inv(P2@L4@P1@L3@L2@L1)
print("Possible L:\n", possible_L)

L = P2@P1@possible_L
print("L:\n", L)

print("Calculated PLU with SciPy\n")
P, L, U = sla.lu(B)

print("P\n", P, '\n', sep='')
print("L\n", L, '\n', sep='')
print("U\n", U, '\n', sep='')
print("PLU\n", P@L@U, sep='')


print('# Exercise 1:')

A = np.array([
    [5, 2, 1],
    [5, 3, 0],
    [-5, -2, -4]])
B = np.array([
    [4],
    [7],
    [8]])
L = np.array([
    [1, 0, 0],
    [1, 1, 0],
    [-1, 0, 1]])
U = np.array([
    [5, 2, 1],
    [0, 1, -1],
    [0, 0, -3]])


print("A\n", A)
print("LU\n", L @ U)
print("B\n", B)
# Backsubstitution to find Y from LY = B
# Y = ss.BackSubstitution(L, B)
Y = sla.solve_triangular(L, B, lower=True)  # Back-Subsitute to solve LY = B
X = sla.solve_triangular(U, Y)  # Back-Subsitute to solve UX = Y

print("AX:\n", A @ X)
print("LUX:\n", L @ U @ X)

print('# Exercise 2:')
L = np.array([
    [1, 0, 0],
    [-1, 1, 0],
    [0, -1, 1]])
U = np.array([
    [1, -1, 0],
    [0, 1, -1],
    [0, 0, 1]])
B = np.array([
    [2],
    [-3],
    [4]])
A = L @ U
Y = sla.solve_triangular(L, B, lower=True)  # Back-Subsitute to solve LY = B
X = sla.solve_triangular(U, Y)  # Back-Subsitute to solve UX = Y
print("B\n", B)
print("AX:\n", A @ X)
print("LUX:\n", L @ U @ X)

print('# Exercise 4:')
I = np.eye(3)
A = np.array([
    [1, 2, 4],
    [2, 1, 3],
    [1, 0, 2]])
B = np.array([
    [1, 2, 4],
    [2, 1, 3],
    [2, 2, 6]])
C = np.array([
    [1, 2, 4],
    [0, -1, -3],
    [2, 2, 6]])
print("A:\n", A)
print("B:\n", B)
print("C:\n", C)

print('Exercise 4a:')
E = lag.RowAdd(I, 0, 2, 1)
print("EA:\n", E @ A)

print('Exercise 4b:')
F = lag.RowAdd(I, 2, 1, -1)
print("EB:\n", F @ B)

print('Exercise 5:')
A = np.array([
    [2, 1, 1],
    [6, 4, 5],
    [4, 1, 3]])
print("A:\n", A)

print('Exercise 5a:')
# A1 = lag.RowAdd(A, 0, 1, -3)
E1 = lag.RowAdd(I, 0, 1, -3)
# print("A1:\n", A1)
# A2 = lag.RowAdd(A1, 0, 2, -2)
E2 = lag.RowAdd(I, 0, 2, -2)
# print("A2:\n", A2)
# A3 = lag.RowAdd(A2, 1, 2, 1)
E3 = lag.RowAdd(I, 1, 2, 1)
# print("A3:\n", A3)
U = E3 @ E2 @ E1 @ A
print("E3 @ E2 @ E1 @ A:\n", U)

print('Exercise 5b:')
print("E1:\n", E1)
E1_inv = lag.RowAdd(I, 0, 1, 3)
# E1_inv = sla.inv(E1)
E2_inv = lag.RowAdd(I, 0, 2, 2)
# E2_inv = sla.inv(E2)
E3_inv = lag.RowAdd(I, 1, 2, -1)
# E3_inv = sla.inv(E3)
L = E1_inv @ E2_inv @ E3_inv
print("LU:\n", L @ U)

print('Exercise 7:')
I = np.eye(3)
A = np.array([
    [1, 3, 2],
    [-2, -6, 1],
    [2, 5, 7]])

# A1 = lag.RowAdd(A, 0, 1, 2)
E1 = lag.RowAdd(I, 0, 1, 2)
E1_inv = lag.RowAdd(I, 0, 1, -2)
# A2 = lag.RowSwap(A1, 1, 2)
P1 = lag.RowSwap(I, 1, 2)
P1_inv = sla.inv(P1)
P1_inv = P1

# A3 = lag.RowAdd(A2, 0, 1, -2)
E2 = lag.RowAdd(I, 0, 1, -2)
E2_inv = lag.RowAdd(I, 0, 1, 2)
# A4 = lag.RowScale(A3, 2, 1/5)
E3 = lag.RowScale(I, 2, 1/5)
E3_inv = lag.RowScale(I, 2, 5)
# A5 = lag.RowScale(A4, 1, -1)
E4 = lag.RowScale(I, 1, -1)
E4_inv = lag.RowScale(I, 1, -1)

U = np.round(E4 @ E3 @ E2 @ P1 @ E1 @ A)
L_prime = np.round(E1_inv @ P1 @ E2_inv @ E3_inv @ E4_inv)
# L_caluldated = P1 @ sla.inv(E4 @ E3 @ E2 @ P1 @ E1)

print("A:\n", A)
print("U:\n", U)
print("L_prime:\n", L_prime)
# Following correctly proves A=L_prime @ U
print("A - L_primeU = 0", A - (L_prime @ U))

P = P1
# L_prime is correct, but not in the form of a lower triangular matrix
# We fix this is here by applying the permutation matrix P1 again (row swaps)
L = P1 @ L_prime

print("PA - LU = 0 \n", P @ A - (L @ U))

print("Exercise 8:")
I = np.eye(3)
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 10]])

# A1 = lag.RowAdd(A, 0, 1, -4)
E1 = lag.RowAdd(I, 0, 1, -4)
# A2 = lag.RowAdd(A1, 0, 2, -7)
E2 = lag.RowAdd(I, 0, 2, -7)
# A3 = lag.RowAdd(A2, 1, 2, -2)
E3 = lag.RowAdd(I, 1, 2, -2)

U = np.round(E3 @ E2 @ E1 @ A)
L = sla.inv(E3 @ E2 @ E1)
P = I # No row swaps needed

print("PA - LU = 0 \n", P @ A - (L @ U))
P_inv, L, U = sla.lu(A) # using SciPy
print("A - P^1LU = 0 (SciPy convention) \n", A - (P_inv @ L @ U))
