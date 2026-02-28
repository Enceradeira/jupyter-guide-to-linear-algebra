from guide_to_linear_algebra.inner_products import unit_vector, angle
import numpy as np
import guide_to_linear_algebra.laguide as lag

# Exercise 1
# Create a Magnitude function that accepts an nx1 NumPy array
# and returns the magnitude of the vector.


def magnitude(V):
    return np.sqrt(lag.DotProduct(V, V))


T = np.array([1, 1, 1]).reshape(3, 1)

print("Magnitude of T:", magnitude(T))

# Exercise 2
# Generate random 4x1 vectors U, V, W and scalar k.
# Verify the four algebraic properties of the dot product:
# 1. Commutative: U·V = V·U
# 2. Distributive: U·(V + W) = U·V + U·W
# 3. Scalar multiplication: (kU)·V = k(U·V)
# 4. Non-negativity: U·U >= 0
print("\nExercise 2")

U = np.random.randint(-10, 10, (4, 1))
V = np.random.randint(-10, 10, (4, 1))
W = np.random.randint(-10, 10, (4, 1))
k = np.random.randint(-10, 10)

# 1. Commutative
print('U·V = V·U:', lag.DotProduct(U, V) == lag.DotProduct(V, U))
# 2. Distributive
print('U·(V + W) = U·V + U·W:', lag.DotProduct(U, V + W)
      == lag.DotProduct(U, V) + lag.DotProduct(U, W))
# 3. Scalar multiplication
print('(kU)·V = k(U·V):', lag.DotProduct(k * U, V) == k * lag.DotProduct(U, V))
# 4. Non-negativity
print('U·U >= 0:', lag.DotProduct(U, U) >= 0)

# Exercise 3
# Given X and Y, determine:
# (a) The angle between X and Y
print("\nExercise 3")

X = np.array([[3], [-1], [2]])
Y = np.array([[4], [2], [0]])

theta = np.acos(lag.DotProduct(X, Y) / (magnitude(X) * magnitude(Y)))
print("Angle between X and Y (in radians):", theta)


# (b) A vector in R^3 orthogonal to X; verify with computation
R = np.array([[-1/3], [1], [1]])
print("R·X = 0:", lag.DotProduct(R, X) == 0)
# (c) A unit vector Z where Z·Y = -||Y||; verify
print(np.acos(-1))  # is Pi, therefore Z' = -Y

Za = Y * -1
Z = Za / magnitude(Za)
print("Z·Y = -||Y||:", lag.DotProduct(Z, Y) == -magnitude(Y))

# Exercise 5
# Create a function that converts an nx1 vector to a unit vector.
# Test on a random 3x1 vector.
print("\nExercise 5")


R5 = np.random.randint(-10, 10, (3, 1))
print("Random vector R5:", R5.T)
print("Unit vector:", unit_vector(R5).T)
print("Magnitude of unit vector:", magnitude(unit_vector(R5)))

# Exercise 6
# Find Y in R^2 satisfying X·Y = ||Y - X|| where X = [1, 1]^T.
# Determine uniqueness; verify computationally.
print("\nExercise 6")

X6 = np.array([[1], [1]])
# not unique solution, found by inserting Y6 into the equation and solving 
# for the unknown component of Y6
Y6 = np.array([[0], [1]])
print("X·Y = ||Y - X||:", lag.DotProduct(X6, Y6) == magnitude(Y6 - X6))

# Exercise 7
# Show that U = [1, -1, 2]^T and W = [2, 0, -1]^T are orthogonal.
print("\nExercise 7")

U7 = np.array([[1], [-1], [2]])
W7 = np.array([[2], [0], [-1]])
print("cos(theta) =", angle(U7, W7))
print("U and W are orthogonal:", angle(U7, W7) == 0.0)

# Exercise 8
# Given ||X+Y|| = 3 and X·Y = 2, find ||X-Y||.
# Using binomial expansion:
# ||X-Y||^2 = (X-Y)·(X-Y) = X·X - 2(X·Y) + Y·Y
# ||X+Y||^2 = (X+Y)·(X+Y) = X·X + 2(X·Y) + Y·Y = 9
# ||X-Y||^2 = ||X+Y||^2 - 4(X·Y) = 9 - 8 = 1
# ||X-Y|| = 1
print("\nExercise 8")
print("||X-Y|| =", np.sqrt(3**2 - 4 * 2))
