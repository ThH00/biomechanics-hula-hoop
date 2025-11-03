import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from network_computation import compute_functional_network
from matplotlib.animation import FuncAnimation, PillowWriter
from contextlib import redirect_stdout
import io

def data_dict_to_2d_array(data_dict,
                          sensors=['OL','OR','IT','IL','IB'],
                          quantities=['wx','wy','wz'],
                          ntime=None):
    """
    Make a 2d array of nodal time series data for network analysis using `compute_functional_network`.
    Each node has one single-component quantity.

    Returns an array with shape = (ntime, nsensors x nquantities)
    order:
        sensor1 quantity1
        sensor1 quantity2
        sensor1 quantity3
        sensor2 quantity1
        ...
    """
    if ntime is not None:
        return np.array([data_dict[s][q]for s in sensors for q in quantities]).T[:ntime]
    else:
        return np.array([data_dict[s][q]for s in sensors for q in quantities]).T
    
def data_dict_to_3d_array(data_dict,
                          sensors=['OL','OR','IT','IL','IB'],
                          quantities=[['dx','vx'],['dy','vy'],['dz','vz']],
                          ntime=None):
    """
    Make a 3d array of nodal time series data for network analysis using `compute_functional_network`.
    Each node has multiple components.
    
    Returns an array with shape = (ntime, nsensors x nquantities, ncomponents per quantity)
    """ 
    data_array = [[data_dict[s][q] for q in qset] for s in sensors for qset in quantities]
    if ntime is not None:
        return np.array(data_array).transpose(2,0,1)[:ntime]
    else:
        return np.array(data_array).transpose(2,0,1)

def scale_data_array(data_array, scale_overall=True):
    if data_array.ndim == 2:
        if scale_overall:
            component_means = data_array.mean(axis=(0,1))
            component_stds = data_array.std(axis=(0,1))
            return (data_array - component_means) / component_stds
        else:
            ntime, nquantities = data_array.shape
            scaled_data_array = np.empty_like(data_array)
            for q in range(nquantities):
                component = data_array[:,q]
                component_mean = component.mean()
                component_std = component.std()
                scaled_data_array[:,q] = (data_array[:,q] - component_mean) / component_std
            return scaled_data_array

    if data_array.ndim == 3:
        if scale_overall:
            component_means = data_array.mean(axis=(0,1))
            component_stds = data_array.std(axis=(0,1))
            return (data_array - component_means) / component_stds
        else:
            ntime, nquantities, ncomponents = data_array.shape
            scaled_data_array = np.empty_like(data_array)
            for q in range(nquantities):
                for c in range(ncomponents):
                    component = data_array[:,q,c]
                    component_mean = component.mean()
                    component_std = component.std()
                    scaled_data_array[:,q,c] = (data_array[:,q,c] - component_mean) / component_std
            return scaled_data_array

def plot_network(C_xys, mapping, target_nodes, width_scale=5.0, self_loops=False):
    if not self_loops:
        np.fill_diagonal(C_xys, 0)

    G = nx.DiGraph(C_xys)

    LG = nx.relabel_nodes(G, mapping)
    pos = nx.circular_layout(LG)

    # Initialize lists for the two groups of edges
    special_edgelist = []
    special2_edgelist = []
    other_edgelist = []
    special_widths = []
    special2_widths = []
    other_widths = []

    # Iterate over all edges in the renamed graph LG
    for u, v, data in LG.edges(data=True):
        weight = data.get('weight', 0)
        
        # Only consider edges with positive weight for drawing
        if weight > 0:
            width = weight * width_scale
            
            # Check if either the source (u) or target (v) is in the target_nodes list
            if u in target_nodes:
                special_edgelist.append((u, v))
                special_widths.append(width)
            elif v in target_nodes:
                special2_edgelist.append((u, v))
                special2_widths.append(width)
            else:
                other_edgelist.append((u, v))
                other_widths.append(width)


    # --- 3. Draw the Network ---
    plt.figure(figsize=(10, 10))

    # 3a. Draw ALL Nodes and Labels
    nx.draw_networkx_nodes(
        LG, 
        pos, 
        node_size=800, 
        node_color='lightgreen', 
        edgecolors='black'
    )
    nx.draw_networkx_labels(LG, pos, font_size=12, font_color='black')


    # 3b. Draw the SPECIAL Edges (Red/Orange color)
    # Draw these first so they are not fully covered by other edges
    nx.draw_networkx_edges(
        LG, 
        pos, 
        edgelist=special_edgelist,
        width=special_widths, 
        edge_color='red', # Highlight color for special edges
        alpha=0.7,
        arrowsize=15, 
        connectionstyle='arc3,rad=0.1' 
    )

    nx.draw_networkx_edges(
        LG, 
        pos, 
        edgelist=special2_edgelist,
        width=special2_widths, 
        edge_color='blue', # Highlight color for special edges
        alpha=0.5,
        arrowsize=10, 
        connectionstyle='arc3,rad=0.1' 
    )

    # 3c. Draw the OTHER Edges (Default color)
    nx.draw_networkx_edges(
        LG, 
        pos, 
        edgelist=other_edgelist,
        width=other_widths, 
        edge_color='gray', # Default color for other edges
        alpha=0.5,
        arrowsize=10, # Slightly smaller arrow/width for background edges
        connectionstyle='arc3,rad=0.1'
    )

    plt.title("Network Visualization: Highlighting Connections to hoop nodes", fontsize=14)
    plt.axis('off') 
    plt.show()

    return plt.gcf()


def animate_networks(data, 
                     rr, C_threshold, T_threshold, n,
                     mapping, target_nodes,
                     width_scale = 5.0,
                     animation_filename = 'network_evolution.gif',
                     window_size=100,
                     self_loops=False):
    # --- ANIMATION PARAMETERS ---
    TOTAL_TIME_POINTS = data.shape[0]
    WINDOW_SIZE = window_size    # Number of time points in the sliding window
    STEP_SIZE = 5                # How much the window shifts per frame (fewer steps = faster animation)
    FPS = 10                     # Frames per second for the final GIF

    n_nodes = data.shape[1]

    # Calculate the start indices for each frame
    start_indices = np.arange(0, TOTAL_TIME_POINTS - WINDOW_SIZE, STEP_SIZE)
    NUM_FRAMES = len(start_indices)

    # --- GLOBAL PLOTTING CONFIGURATION (from your code) ---
    network_titles = ['C_xys Network', 'C_yxs Network', 'T_xys Network', 'T_yxs Network']

    # Prepare the figure and axes once
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes_flat = axes.flatten()

    # Pre-calculate the fixed circular layout (should not change over time)
    # We use a dummy graph just to get the node ordering for the layout.
    dummy_G = nx.DiGraph(np.zeros((n_nodes, n_nodes)))
    dummy_LG = nx.relabel_nodes(dummy_G, mapping)
    fixed_pos = nx.circular_layout(dummy_LG)

    # Define a single function to draw a network on a specific axis
    def draw_single_network(ax, network_data, title, pos, t_start):
        """Draws a single network plot on the given axis."""
        
        # Clear the previous drawing
        ax.clear()
        ax.axis('off') # Keep axis off after clearing
        
        # --- Graph Creation and Relabeling ---
        G = nx.DiGraph(network_data)
        LG = nx.relabel_nodes(G, mapping)

        # --- Separate Edges for Coloring ---
        special_edgelist = []
        special2_edgelist = []
        other_edgelist = []
        special_widths = []
        special2_widths = []
        other_widths = []

        for u, v, data in LG.edges(data=True):
            weight = data.get('weight', 0)
            
            if weight > 0:
                width = weight * width_scale
                is_from_target = u in target_nodes
                is_to_target = v in target_nodes
                
                # Grouping logic (simplified)
                if is_from_target:
                    special_edgelist.append((u, v))
                    special_widths.append(width)
                elif is_to_target:
                    special2_edgelist.append((u, v))
                    special2_widths.append(width)
                else:
                    other_edgelist.append((u, v))
                    other_widths.append(width)

        # --- Draw the Network ---
        
        # 1. Draw Nodes and Labels
        nx.draw_networkx_nodes(LG, pos, node_size=700, node_color='lightgreen', edgecolors='black', ax=ax)
        nx.draw_networkx_labels(LG, pos, font_size=10, font_color='black', ax=ax)

        # 2. Draw Special Edges (FROM target_nodes - Red)
        nx.draw_networkx_edges(
            LG, pos, edgelist=special_edgelist, width=special_widths, 
            edge_color='red', alpha=0.7, arrowsize=15, 
            connectionstyle='arc3,rad=0.1', ax=ax
        )

        # 3. Draw Special2 Edges (TO target_nodes - Blue)
        nx.draw_networkx_edges(
            LG, pos, edgelist=special2_edgelist, width=special2_widths, 
            edge_color='blue', alpha=0.7, arrowsize=15, 
            connectionstyle='arc3,rad=0.1', ax=ax
        )

        # 4. Draw Other Edges (Gray)
        nx.draw_networkx_edges(
            LG, pos, edgelist=other_edgelist, width=other_widths, 
            edge_color='gray', alpha=0.5, arrowsize=10, 
            connectionstyle='arc3,rad=0.1', ax=ax
        )

        # Set Title and time annotation
        ax.set_title(f"{title}\nTime Window: {t_start}-{t_start + WINDOW_SIZE}", fontsize=12)

    # --- 2. The Update Function for the Animation ---
    def update(frame_index):
        """
        Function called by FuncAnimation for each frame.
        It calculates the networks for a new time window and updates the plots.
        """
        t_start = start_indices[frame_index]
        t_end = t_start + WINDOW_SIZE
        
        # 1. Slice the data for the current window
        data_window = data[t_start:t_end, :]

        # 2. Calculate the four networks
        # The output is (C_xys, C_yxs, T_xys, T_yxs) as per your setup
        with redirect_stdout(io.StringIO()): # suppress print statements
            G, G_, common_G, T_diff, C_diff, C_xys, C_yxs, T_xys, T_yxs = compute_functional_network(
                data_window, rr, C_threshold=C_threshold, T_threshold=T_threshold, n=n
            )
        
        if not self_loops:
            np.fill_diagonal(C_xys, 0)

        networks_data = [C_xys, C_yxs, T_xys, T_yxs]

        # 3. Redraw all four subplots
        for i in range(4):
            ax = axes_flat[i]
            network_data = networks_data[i]
            title = network_titles[i]
            
            draw_single_network(ax, network_data, title, fixed_pos, t_start)
        
        # Return the updated artists (necessary for FuncAnimation)
        return axes_flat

    # --- 3. Run and Save the Animation ---
    print(f"Generating animation with {NUM_FRAMES} frames...")

    anim = FuncAnimation(
        fig, 
        update, 
        frames=NUM_FRAMES,
        blit=False,  # Set to False, as NetworkX often doesn't handle blitting well
        interval=1000/FPS # Delay between frames in ms
    )

    # Save the animation as a GIF
    writer = PillowWriter(fps=FPS)
    anim.save(animation_filename, writer=writer)

    plt.close(fig) # Close the figure to free up memory
    print(f"Animation saved as {animation_filename}")