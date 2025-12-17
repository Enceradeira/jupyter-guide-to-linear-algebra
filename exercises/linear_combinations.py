import numpy as np
import guide_to_linear_algebra.laguide as lag

# Exercise 1

X = np.array([[-1], [5], [-6]])
V = np.array(
    [[2, 2, 2],
     [0, 4, -12],
     [7, 5, 13],
     ])

VX = np.hstack((V, X))
R = lag.FullRowReduction(VX)
print("VX =", VX)
print("Reduced form VX:", "\n", R)

c3 = 1
c2 = 1.25 + 3 * c3
c1 = -1.75 - 4 * c3

C = np.array([c1, c2, c3])
print("infinite solutions, one is C =", C)
print("Check V * C =", V @ C, "\n")

# Exercise 2
X = np.array([[1], [3], [-1]])
V = np.array([[1, 2, 2],
              [0, -2, 0],
              [0, 1, 4]])
VX = np.hstack((V, X))
R = lag.FullRowReduction(VX)
print("VX =", VX)
print("Reduced form VX:", "\n", R)
print("Yes, X lies in the span of the vectors in V, since all rows have a pivot, therefore is one solution for C given X\n")

# Exercise 3
X = np.array([
    [1, 1, 2, 3],
    [1, 0, 0, 0],
    [1, 0, 1, 1],
    [1, 2, 1, 2]])
R = lag.FullRowReduction(X)
print("Reduced form X:", "\n", R)
print("X lies in R4 since there is a pivot in every row\n. Any C would have a solution for. C X = Y")

# Exercise 4
W = np.array([
    [2, 3, 4],
    [3, 0, -3],
    [-1, 1, 3]])

# a.) Is B in the span of the columns of W?
B = np.array([[1], [1], [1]])
WB = np.hstack((W, B))
R = lag.FullRowReduction(WB)
print("Reduced form WB:", "\n", R)
print("B is not in spam, there is no C for with W C = B, since there is no solution for the system\n")
# b.) Is C in the span of W
C = np.array([[3], [0], [1]])
WC = np.hstack((W, C))
R = lag.FullRowReduction(WC)
print("Reduced form WC:", "\n", R)
print("C is in spam, there are infinite X such that W X = C\n")
# c.) Find a nonezero V in the span of W. Given V = [v1, v2, v3] | v1 = 0
# W C = V
R = lag.FullRowReduction(W)
print("Reduced form W:", "\n", R)

V = np.array([[0], [9], [-5]])
C = np.array([2, 0, -1])
print("V = W C =", W @ C, "\n")

WV = np.hstack((W, V))
R = lag.FullRowReduction(WV)
print("Reduced form WV:", "\n", R)

# d.) Is the span {W1, W2, W3} and {V1, V2}. Is {V1, V2} in the span of W?
# For V1:
V1 = np.array([[8], [-3], [5]])

# Is there a solution for W C = V1?
WV1 = np.hstack((W, V1))
R = lag.FullRowReduction(WV1)
print("Reduced form WV1:", "\n", R)

print("No, there is no solution for W C = V1, therefore V1 is not in the span of W\n")

# Exercise 5
A = np.array([
    [1, 1, 3],
    [2, 0, 1],
    [3, 1, 1]])
X = np.array([[1], [1], [1]])
AX = np.hstack((A, X))
R = lag.FullRowReduction(AX)
print("Reduced form AX:", "\n", R)
print("Yes, X is in the span of the columns of A, since there is a solution for A C = X\n")

# Find another vetcotr in the spane of A -> any vector on R3 is in the span
X2 = np.array([[3], [-3], [7]])
AX2 = np.hstack((A, X2))
R = lag.FullRowReduction(AX2)
print("Reduced form AX2:", "\n", R)
print("Yes, X2 is in the span of the columns of A, since there is a solution for A C = X2\n")

# Exercise 6
R = np.array([
    [1, 1, 0, -1],
    [1, 1, 0, 1],
    [-1, -1, 1, -1],
    [1, 1, -2, 0]])

# Find V in the span, therefore R C = V has a solution

R = lag.FullRowReduction(R)
print("Reduced form R:", "\n", R)

C = np.array([1, 1, 1, 0])
V = R @ C
print("V = R C =", V, "\n")
print("V is in the span of R since was calculated as R C for some C\n")

V2 = np.array([[1], [1], [1], [1]])
RV2 = np.hstack((R, V2))
R = lag.FullRowReduction(RV2)
print("Reduced form RV2:", "\n", R)
print("V2 is not in the span of R since there is no solution for R C = V2\n")


# Exercise 7
# Find V3 that is not in the span of {V1, V2}

V = np.array([
    [1, 2],
    [1, 0],
    [1, 4]])

R = lag.FullRowReduction(V)
print("Reduced form V:", "\n", R)
V3 = np.array([[1], [1], [0]])

VV3 = np.hstack((V, V3))
R = lag.FullRowReduction(VV3)
print("Reduced form VV3:", "\n", R)
print("V3 is not in the span of V since there is no solution for V C = V3\n")

# Exercise 8
# Because B could have a pivot, an therefore not hava a solution

# Exercise 9
a = -1  # Does span R3
V = np.array([
    [1, 4, 1],
    [2, 5, 0],
    [3, 6, a]])

R = lag.FullRowReduction(V)
print("Reduced form V:", "\n", R)
