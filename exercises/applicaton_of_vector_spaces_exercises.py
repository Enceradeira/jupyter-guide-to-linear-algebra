import numpy as np
import guide_to_linear_algebra.laguide as lag

# Exercise 1:

# Given: Formula and elements,dimensions of C,H and O
# Find: Combination of C, H and O that gives the formula
# Solution:

C1 = np.array([3, 8, 0]).reshape(-1, 1)  # C3H8
C2 = np.array([0, 0, 2]).reshape(-1, 1)  # O2
# equal
C3 = np.array([1, 0, 2]).reshape(-1, 1)  # CO2
C4 = np.array([0, 2, 1]).reshape(-1, 1)  # H2O

M = np.hstack((C1, C2, -C3, -C4))
M_reduced = lag.FullRowReduction(M)
print("Reduced form of the matrix M:", "\n", M_reduced, "\n")

# x4 is a free variable, let x4 = t
# x4 = t
# x3 = 0.75t
# x2 = 1.25t
# x1 = 0.25t

t = 4  # Choose t = 4 to get integer coefficients

# Therefore, the combination is:
x1 = int(0.25 * t)  # Coefficient for C3H8
x2 = int(1.25 * t)  # Coefficient for O2
x3 = int(0.75 * t)  # Coefficient for CO2
x4 = int(t)         # Coefficient for H2O

print(f"The combination is: {x1} C3H8 + {x2} O2 -> {x3} CO2 + {x4} H2O\n")

# Exercise 2:

# Given: Formulate and elements, dimensions of Al, H, SO4
# Find: Combination of Al, H, SO4 that gives the formula
# Solution:

C1 = np.array([1, 0, 0]).reshape(-1, 1)  # Al
C2 = np .array([0, 2, 1]).reshape(-1, 1)  # H2SO4
# equal
C3 = np.array([2, 0, 3]).reshape(-1, 1)  # Al2(SO4)3
C4 = np.array([0, 2, 0]).reshape(-1, 1)  # H2

M = np.hstack((C1, C2, -C3, -C4))
M_reduced = lag.FullRowReduction(M)
print("Reduced form of the matrix M for Exercise 2:", "\n", M_reduced, "\n")

# x4 is a free variableo
t = 3  # Choose t = 3 to get integer coefficients

x4 = int(t)         # Coefficient for H2
x3 = int(1/3 * t)  # Coefficient for Al2(SO4)3
x2 = int(t)        # Coefficient for H2SO4
x1 = int(2/3 * t)  # Coefficient for Al

print(f"The combination is: {x1} Al + {x2} H2SO4 -> {x3} Al2(SO4)3 + {x4} H2\n")


# Exercise 3:

# Given: Formulate and elements, dimensions of Ag, H, S, O
# Find: Combination of Ag, H, S, O that gives the formula
# Solution:

C1 = np.array([1, 0, 0, 0]).reshape(-1, 1)  # Ag
C2 = np.array([0, 2, 1, 0]).reshape(-1, 1)  # H2SO
C3 = np.array([0, 0, 0, 2]).reshape(-1, 1)  # O2
# equal
C4 = np.array([2, 0, 1, 0]).reshape(-1, 1)  # Ag2SO
C5 = np.array([0, 2, 0, 1]).reshape(-1, 1)  # H2O

M = np.hstack((C1, C2, C3, -C4, -C5))
M_reduced = lag.FullRowReduction(M)
print("Reduced form of the matrix M for Exercise 3:", "\n", M_reduced, "\n")

# x5 is a free variableo
t = 2  # Choose t = 2 to get integer coefficients

x5 = int(t)
x4 = int(t)
x3 = int(0.5 * t)
x2 = int(t)
x1 = int(2 * t)

print(f"The combination is: {x1} Ag + {x2} H2S + {x3} O2 -> {x4} Ag2S + {x5} H2O\n")
