import numpy as np
import random
import guide_to_linear_algebra.laguide as lag
import guide_to_linear_algebra.plt_helper as ph

# Finding number of paths of length 3 between nodes in a graph
B = np.array([[0, 1, 1, 0, 0, 0],
              [1, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 1],
              [1, 1, 0, 0, 1, 0],
              [0, 1, 0, 1, 0, 0,],
              [0, 0, 0, 0, 1, 0]])
print("Given adjacency matrix:\n", B)

L = B@B@B

print("Number of paths of length 3 between nodes:\n", L)

# Finding cliques
A = np.array([[0, 1, 1, 1, 0, 0],
              [1, 0, 1, 1, 0, 0],
              [0, 0, 0, 0, 1, 1],
              [1, 1, 0, 0, 1, 0],
              [0, 1, 0, 1, 0, 0,],
              [0, 0, 0, 0, 1, 0]])
print("Adjacency matrix:", "\n", A)


def select_bidirectional_edges(A):
    """Return a matrix with 1s only where edges exist in both directions."""
    return np.logical_and(A, A.T).astype(int)


C = select_bidirectional_edges(A)

print("Nodes connected both ways:")
print(np.array2string(C, separator=' '))
print("Nodes connected both ways: C=", "\n", C)
print("Nodes connected both ways, with number of 3 paths: C@C@C=", "\n", C@C@C)

# node_positions = lag.DrawGraph(A)
# lag.HighlightSubgraph(A, node_positions, [0, 1, 3])
# ph.open_plt()

# Finding clicques in random graphs
N = 10
R = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if (random.random() > 0.5 and i != j):
            R[i, j] = 1

big_graph_positions = lag.DrawGraph(R)

C = select_bidirectional_edges(R)

C_3 = C@C@C


def select_nodes_having_path_length_3_to_themselves(A, N):
    """Select nodes that have a path of length 3 to themselves."""
    clique_nodes = []
    for i in range(N):
        if (A[i, i] != 0):
            clique_nodes.append(i)
    return clique_nodes


clique_nodes = select_nodes_having_path_length_3_to_themselves(C_3, N)

print(C_3)
lag.HighlightSubgraph(R, big_graph_positions, clique_nodes)
ph.open_plt()
