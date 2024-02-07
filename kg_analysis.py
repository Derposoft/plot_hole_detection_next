import os
import numpy as np
import pickle as pkl
import time

# Utility fn
lasttime = time.time()
def checktime(string="time check"):
    global lasttime
    currtime = time.time()
    print(f"{currtime - lasttime:0.4f}s -- {string}")
    lasttime = currtime


# Utility fn 2
def count_connected_graphs(edges, n) -> tuple[int, list[int]]:
    # Create an adjacency list from the given edges
    adj = {i: [] for i in range(n)}
    for u, v in zip(*edges):
        u, v = u.item(), v.item()
        adj[u].append(v)
        adj[v].append(u)

    # DFS to find connected components
    visited = set()
    n_components = 0
    sizes = []
    def dfs(node):
        visited.add(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                dfs(neighbor)
    for node in range(n):
        if node not in visited:
            dfs(node)
            n_components += 1
            sizes.append(len(visited) - sum(sizes))
    return n_components, sizes


# Load data
kg_file = "FicClaim/FicClaim-1-error.pkl"
with open(kg_file, "rb") as f:
    dataset = pkl.load(f)
checktime("dataset loading")

# Analyze kgs in data
NODE_FEATS = "node_feats"
EDGE_INDICES = "edge_indices"
EDGE_FEATS = "edge_feats"
total_kgs = len(dataset)
total_nodes = 0
total_edges = 0
total_connected_pieces = 0
connected_pieces = []
for i, (_, _2, kg, _3) in enumerate(dataset):
    total_nodes += len(kg[NODE_FEATS])
    total_edges += len(kg[EDGE_FEATS])
    connected_graphs, connected_graph_sizes = count_connected_graphs(kg[EDGE_INDICES], len(kg[NODE_FEATS]))
    total_connected_pieces += connected_graphs
    connected_pieces += connected_graph_sizes
    if i % 100 == 0:
        print(".", end='')
print("\n")

checktime("dataset analysis")

# Print analysis
print(f"Total number of nodes: {total_nodes}")
print(f"Total number of edges: {total_edges}")
print(f"Total number of connected pieces: {total_connected_pieces}")
print(f"Total number of kgs: {total_kgs}")

print(f"Avg nodes/kg: {total_nodes / total_kgs}")
print(f"Avg edges/kg: {total_edges / total_kgs}")
print(f"Avg connected graphs/kg: {total_connected_pieces / total_kgs}")
print(f"Median size of connected graphs: {np.median(connected_pieces)}")
print(f"Max size of connected graphs: {np.max(connected_pieces)}")
print(f"Mean size of connected graphs: {np.mean(connected_pieces)}")
print(f"Avg edges/node: {total_edges / total_nodes}")