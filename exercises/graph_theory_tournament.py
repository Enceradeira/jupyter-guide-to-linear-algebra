import numpy as np
import random
import guide_to_linear_algebra.laguide as lag
import guide_to_linear_algebra.plt_helper as ph

D = np.array([
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 1, 0, 1, 1],
    [1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0]])
print("which player won agains which? D=\n", D)
print("which player won agaist one that bet another? D@D=\n", D@D)
print("D + D@D=\n", D + D@D)
positions = lag.DrawGraph(D)
ph.open_plt()
