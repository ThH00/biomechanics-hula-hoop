import numpy as np
from pyunicorn.timeseries import InterSystemRecurrenceNetwork
import networkx as nx
import plotly.graph_objects as go

# --- Your original data ---
time_steps = 1000
dimensions = 6

sensor_a_data = np.random.rand(time_steps, dimensions) + np.sin(np.linspace(0, 10 * np.pi, time_steps))[:, np.newaxis]
sensor_b_data = np.random.rand(time_steps, dimensions) + np.cos(np.linspace(0, 10 * np.pi, time_steps))[:, np.newaxis]

network = InterSystemRecurrenceNetwork(
    sensor_a_data,
    sensor_b_data,
    recurrence_rate=(0.05, 0.05, 0.05),
    metric="euclidean"
)

print("Number of links:", network.n_links)
print("Average path length:", network.average_path_length())

# --- Get adjacency & graph ---
adj_matrix = network.adjacency
n_nodes = adj_matrix.shape[0]
n_A = sensor_a_data.shape[1]   # dimensions for A

G = nx.from_numpy_array(adj_matrix)

# --- 3D layout ---
pos = nx.spring_layout(G, dim=3, seed=42)  # dict: node -> (x,y,z)

# Extract node coordinates
x_nodes = [pos[i][0] for i in range(n_nodes)]
y_nodes = [pos[i][1] for i in range(n_nodes)]
z_nodes = [pos[i][2] for i in range(n_nodes)]

# Extract edges as line segments
edge_x, edge_y, edge_z = [], [], []
for e in G.edges():
    edge_x.extend([pos[e[0]][0], pos[e[1]][0], None])
    edge_y.extend([pos[e[0]][1], pos[e[1]][1], None])
    edge_z.extend([pos[e[0]][2], pos[e[1]][2], None])

# --- Build Plotly figure ---
edge_trace = go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    mode='lines',
    line=dict(color='lightgray', width=1),
    hoverinfo='none'
)

node_trace = go.Scatter3d(
    x=x_nodes, y=y_nodes, z=z_nodes,
    mode='markers',
    marker=dict(
        size=5,
        color=['red' if i < n_A else 'blue' for i in range(n_nodes)],
        opacity=0.8
    ),
    hovertext=[f"Node {i}" for i in range(n_nodes)],
    hoverinfo='text'
)

fig = go.Figure(data=[edge_trace, node_trace])
fig.update_layout(
    title="Inter-System Recurrence Network (3D)",
    showlegend=False,
    margin=dict(l=0, r=0, t=40, b=0),
    scene=dict(xaxis=dict(showbackground=False),
               yaxis=dict(showbackground=False),
               zaxis=dict(showbackground=False))
)
fig.show()
