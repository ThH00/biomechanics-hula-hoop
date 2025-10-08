# import numpy as np
# from network_computation import network_computation, compute_functional_network
# 
# # Load data
# data = np.load("data.npy")
 
# # 
# # G, G_, common_G = compute_functional_network(data, rr=(0.03, 0.03, 0.02), n=45)

#   network_computation(data_directory_main,
#                         result_directory_main,
#                         time_series_length=25,
#                         noise_level=1)

# print('done')


import networkx as nx
import matplotlib.pyplot as plt

# Read edge list
G = nx.read_edgelist(
    "/Users/theresahonein/Desktop/network-methods-code/2024-code-ExploringLocalizationInNonlinearOscillators/results/all_time_series/1.0/data/G_cartesian", 
    create_using=nx.DiGraph(),  # directed graph
    nodetype=int                 # convert node IDs to integers
)

# Draw the network
plt.figure(figsize=(8,6))
pos = nx.spring_layout(G, seed=42)  # layout for nice spacing
nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", arrows=True)
plt.show()


print("Nodes:", G.nodes())
print("Edges:", G.edges())
print("Number of strongly connected components:", nx.number_strongly_connected_components(G))
print("In-degree of node 1:", G.in_degree(1))
print("Out-degree of node 1:", G.out_degree(1))
