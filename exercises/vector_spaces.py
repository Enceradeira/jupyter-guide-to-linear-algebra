import matplotlib.pyplot as plt
import numpy as np
import guide_to_linear_algebra.laguide as lag
import guide_to_linear_algebra.plt_helper as ph

fig, ax = plt.subplots()
options = {"head_width": 0.1, "head_length": 0.2, "length_includes_head": True}

ax.arrow(0, 0, 1, 3, fc='b', ec='b', **options)

ax.text(1, 2, '$U_1$')

ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_aspect('equal')
ax.grid(True, ls=':')

# ph.open_plt()

# reduced row echelon form
C = np.random.randint(-2, 3, size=(3, 5))
C_full_reduced = lag.FullRowReduction(C)
C_reduced = lag.RowReduction(C)

print(C, '\n')
print(C_reduced, '\n')
print(C_full_reduced)

B_augmented = np.array(
    [[-2, 2, -2, 2, 0],
     [1, -2, -2, 0, -1],
     [1, 0, 2, -2, 1]])
B_augmented_reduced = lag.FullRowReduction(B_augmented)
print(B_augmented_reduced)
