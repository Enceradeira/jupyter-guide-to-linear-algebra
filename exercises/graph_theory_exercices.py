import numpy as np
import random
import guide_to_linear_algebra.laguide as lag
import guide_to_linear_algebra.plt_helper as ph

# Exercise 1: Create you own adjacency graph
J1 = np.array([
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0]])
# positions = lag.DrawGraph(J1)
# ph.open_plt()

# Excercise 2
J2 = np.array([
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0]])

# lag.DrawGraph(J2, positions)
# ph.open_plt()

# Exercise 3
N = 10
R = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if (random.random() > 0.5 and i != j):
            R[i, j] = 1

big_graph_positions = lag.DrawGraph(R)

C = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if (R[i, j] == 1 and R[j, i] == 1):
            C[i, j] = 1

print("C:\n", C)
C_3 = C@C@C
clique_nodes = []
for i in range(N):
    if (C_3[i, i] != 0):
        clique_nodes.append(i)
print("C3:\n", C_3)
lag.HighlightSubgraph(R, big_graph_positions, clique_nodes)
print("Clique nodes:", clique_nodes)

ph.open_plt()


def find_cliques(C, clique_nodes):
    """
    Separate clique nodes into distinct cliques.

    Parameters:
    C: adjacency matrix with only bidirectional edges
    clique_nodes: list of nodes that are part of at least one clique

    Returns:
    List of lists, where each inner list contains nodes of one clique
    """
    cliques = []
    remaining_nodes = clique_nodes.copy()

    while remaining_nodes:
        # Start a new clique with the first remaining node
        current_node = remaining_nodes[0]
        current_clique = [current_node]
        remaining_nodes.remove(current_node)

        # Find all nodes connected bidirectionally to all nodes in current_clique
        for node in remaining_nodes:  # Use slice to iterate over copy
            # Check if node is connected to all nodes in current_clique
            connected_to_all = True
            for clique_member in current_clique:
                if C[node, clique_member] == 0 or C[clique_member, node] == 0:
                    connected_to_all = False
                    break

            if connected_to_all:
                current_clique.append(node)
                # remaining_nodes.remove(node)

        # Only add cliques with at least 3 nodes
        if len(current_clique) >= 3:
            cliques.append(sorted(current_clique))

    return cliques


# Use the function with the generated C matrix and clique_nodes
# breakpoint()
if len(clique_nodes) >= 1:
    separated_cliques = find_cliques(C, clique_nodes)
    print("Cliques found:")
    for i, clique in enumerate(separated_cliques):
        print(f"Clique {i+1}: {clique}")
else:
    print("No cliques found.")
