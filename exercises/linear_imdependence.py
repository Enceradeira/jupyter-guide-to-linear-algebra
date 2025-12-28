import numpy as np
import guide_to_linear_algebra.laguide as lag

# Exercise 1:
U = np.array([
    [0, 1, 3],
    [5, -1, 2],
    [2, 0, 2],
    [2, -1, -1]])
# ZERO=np.array([[0], [0], [0], [0]])

R = lag.FullRowReduction(U)
print("Reduced form U:", "\n", R)
print('U is linearly dependent since there is no pivot in every column\n')

# Exercise 2:
W = np.array([
    [1, 0, -1, 1],
    [0, 1, 0, 1],
    [0, -1, -1, 1],
    [1, 0, 0, -1]])

ZERO = np.array([[0], [0], [0], [0]])
WZERO = np.hstack((W, ZERO))

R = lag.FullRowReduction(WZERO)
print("Reduced form WZERO:", "\n", R)
print('W is linearly depnendent since there is no pivot in every column\n')

WLeft = W[:, :3]
Wother = W[:, 3:]
print("WLeft =", WLeft)
print("Wother =", Wother)

# WLeft * C = Wother
R = lag.FullRowReduction(W)
print("Reduced form W:", "\n", R)
c1 = -1
c2 = 1
c3 = -2

Combination = c1 * WLeft[:, 0] + c2 * WLeft[:, 1] + c3 * WLeft[:, 2]
print("Wother = Combination =", Combination)

# Exercise 3:
b = 6  # found by manual gauss elimination
X = np.array([
    [1, 1, 2],
    [0, 2, 4],
    [1, 3, b]])
R = lag.FullRowReduction(X)
print("Reduced form X:", "\n", R)

c1 = 0
c2 = 2

X2 = X[:, 2]
Combinaion = c1 * X[:, 0] + c2 * X[:, 1]
print("X2 =", X2)
print("X2 = Combination =", Combinaion)

# Exercise 4:


# Exercise 5

A = np.array([
    [1, 1, 2],
    [2, 0, 1],
    [3, 1, 1]])
R = lag.FullRowReduction(A)
print("Reduced form A:", "\n", R)
print('The columns of A are linearly independent since there is a pivot in every column\n')

# Exercise 6
A = np.array([
    [1, 2, 3],
    [1, 0, 1],
    [1, 1, 2]])

R = lag.FullRowReduction(A)
print("Reduced form A:", "\n", R)

V = np.array([1, 1, -1])
NULL = A @ V
print("A V =", NULL)


# Exercise 7
D = np.array([
    [4, 4, 3],
    [8, 8, 6],
    [1, 0, 1]])

R = lag.FullRowReduction(D)
print("Reduced form D:", "\n", R)

t = 4
V1 = np.array([-t, t/4, t])
t = 8
V2 = np.array([-t, t/4, t])

# Show V1 and V2 are in N(D) null space
print("D V1 =", D @ V1)
print("D V2 =", D @ V2)

# Exercise 8

