import inspect
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from network_computation import compute_functional_network, compute_functional_network_th
from matplotlib.animation import FuncAnimation, PillowWriter
from contextlib import redirect_stdout
import io
from pathlib import Path
import os

SYMDICT = {
    'wx': r"$\omega_{x}$",
    'wy': r"$\omega_{y}$",
    'wz': r"$\omega_{z}$",
    'wxy': r"$\omega_{xy}$",
    'dx': r"$d_{x}$",
    'dy': r"$d_{y}$",
    'dz': r"$d_{z}$",
    'vx': r"$v_{x}$",
    'vy': r"$v_{y}$",
    'vz': r"$v_{z}$",
    'phi': r"$\phi$",
    'theta': r"$\theta$",
    'psi': r"$\psi$",
    'phidot': r"$\dot{\phi}$",
    'thetadot': r"$\dot{\theta}$",
    'psidot': r"$\dot{\psi}$",
}

SENSOR_DICT = {
    'OR': 'h',
    'IB': 'f',
    'IT': 't',
    'IL': 'm'
}

SENSOR_DICT_LONG = {
    'OR': 'hoop',
    'IB': 'femur',
    'IT': 'tibia',
    'IL': 'metatarsal'
}

LONG_DICT = {
        f"{n}{symbol}": f"{SENSOR_DICT_LONG[s]} {symbol}"  # e.g. "femur $\phi$")
        for s,n in SENSOR_DICT.items()
        for name,symbol in SYMDICT.items()
    }

NET_DICT = {
    'C_xys': r"$C_{xy}$",
    'T_xys': r"$T_{xy}$",
    'C_diff': r'$C_{\mathrm{diff}}$',
    'T_diff': r'$T_{\mathrm{diff}}$',
}

def data_to_array_by_quantity(data_dict,
                              quantities={
                                    'OR':[['time','wxy']],
                                    'IB':[['time','wx'], ['time','wy'], ['time','wz']],
                                    'IT':[['time','wx'], ['time','wy'], ['time','wz']],
                                    'IL':[['time','wx'], ['time','wy'], ['time','wz']],
                                },
                              ntime=None):
    """
    Returns an array with shape = (ntime, nquantities, ncomponents per quantity)
    Returns an array with shape = (1894,  10,          2)
    """
    data_array = [[data_dict[s][q] for q in qset] for s,qsets in quantities.items() for qset in qsets]
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

def plot_network(coeff_xys,
                 mapping,
                 target_nodes,
                 width_scale=5.0,
                 self_loops=False,
                 title=None,
                 diffs=False,
                 draw_to_target_edges=True,
                 draw_from_target_edges=True,
                 draw_no_target_edges=True):
    if not self_loops:
        np.fill_diagonal(coeff_xys, 0)

    # By default, the arrows are drawn from x to y.
    # if Cdiff,Tdiff is given (diffs is True),
    # then we must draw the graph based on Cdiff.T and Tdiff.T
    # so that when Cdiff,Tdiff>0, the arrow is drawn from y to x.
    if diffs:
        coeff_yxs = coeff_xys.T
        G = nx.DiGraph(coeff_yxs)
    else:
        G = nx.DiGraph(coeff_xys)

    LG = nx.relabel_nodes(G, mapping)
    pos = nx.circular_layout(LG)

    # Initialize lists for the three groups of edges

    target_x_edges = []
    target_y_edges = []
    no_target_edges = []

    target_x_widths = []
    target_y_widths = []
    no_target_widths = []

    # Iterate over all edges in the renamed graph LG
    for x, y, data in LG.edges(data=True):
        weight = data.get('weight', 0)
        
        # Only consider edges with positive weight for drawing
        if weight > 0:
            width = weight * width_scale
            
            # Check if either x is a target, or y is a target
            if x in target_nodes:
                target_x_edges.append((x, y))
                target_x_widths.append(width)
            elif y in target_nodes:
                target_y_edges.append((x, y))
                target_y_widths.append(width)
            else:
                no_target_edges.append((x, y))
                no_target_widths.append(width)


    # --- Draw the Network ---
    plt.figure(figsize=(10, 10))

    # Draw ALL Nodes and Labels
    nx.draw_networkx_nodes(
        LG, 
        pos, 
        node_size=2500, 
        node_color='lightgreen', 
        edgecolors=None
    )
    nx.draw_networkx_labels(LG, pos, font_size=12, font_color='black')


    # Draw the edges

    if draw_from_target_edges:
        # Edges where target is x (y if diffs is True, because Cdiff.T,Tdiff.T are used).
        # For Cdiff,Tdiff, a positive value means that X -> Y (the target affects other nodes).
        # For Cxy,Txy, a positive value means Cxy,Txy>0 with target as x.
        arrows = nx.draw_networkx_edges(
            LG, 
            pos, 
            edgelist=target_x_edges,
            width=target_x_widths, 
            edge_color='red',
            alpha=0.5 if diffs else 1.0,
            arrowsize=10, 
            connectionstyle='arc3,rad=0.1',
            min_target_margin=25
        )
        for arrow in arrows:
            arrow.set_joinstyle('miter')
            arrow.set_capstyle('butt')

    if draw_to_target_edges:
        # Edges where target is y (x if diffs is True, because Cdiff.T,Tdiff.T are used).
        # For Cdiff,Tdiff, a positive value means that Y -> X (other nodes affect the target).
        # For Cxy,Txy, a positive value means Cxy,Txy>0 with target as y.
        arrows = nx.draw_networkx_edges(
            LG, 
            pos, 
            edgelist=target_y_edges,
            width=target_y_widths, 
            edge_color='black' if diffs else 'blue',
            alpha=1.0,
            arrowsize=10, 
            connectionstyle='arc3,rad=0.1',
            min_target_margin=25
        )
        for arrow in arrows:
            arrow.set_joinstyle('miter')
            arrow.set_capstyle('butt')


    if draw_no_target_edges:
        # Edges that don't involve the target nodes
        arrows = nx.draw_networkx_edges(
            LG, 
            pos, 
            edgelist=no_target_edges,
            width=no_target_widths, 
            edge_color='gray',
            alpha=0.5,
            arrowsize=10, 
            connectionstyle='arc3,rad=0.1',
            min_target_margin=25
        )
        for arrow in arrows:
            arrow.set_joinstyle('miter')
            arrow.set_capstyle('butt')

    lim = 1.1
    plt.ylim(-lim, lim) 
    plt.xlim(-lim, lim)
    plt.title(title, fontsize=14)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off') 

    return plt.gcf()


def get_time_networks(data,
                      use_thresholds=False,
                      th=(0.1, 0.1, 0.05),
                      rr=(0.06,0.06,0.02),
                      C_threshold=0.02,
                      T_threshold=0.02,
                      window_size=100,
                      step_size=5,
                      sandwiched_couples=False,
                    ):
    n_time = data.shape[0]
    n_nodes = data.shape[1]
    start_indices = np.arange(0, n_time - window_size, step_size)
    n_frames = len(start_indices)

    def get_data_window(frame_index,start_indices,window_size):
        """Slice the data for the current window"""
        t_start = start_indices[frame_index]
        t_end = t_start + window_size
        data_window = data[t_start:t_end, :]
        return data_window

    def get_network_window(data_window,n=n_nodes):
        """Get the networks data for the current window"""
        # Calculate the four networks
        with redirect_stdout(io.StringIO()): # suppress print statements
            if use_thresholds:
                G, G_, common_G, C_xys, C_yxs, T_xys, T_yxs, rrx, rrxy = compute_functional_network_th(
                    data_window, th=th, n=n, sandwiched_couples=sandwiched_couples)
            else:
                G, G_, common_G, T_diff, C_diff, C_xys, C_yxs, T_xys, T_yxs = compute_functional_network(
                    data_window, rr=rr, C_threshold=C_threshold, T_threshold=T_threshold,
                    n=n,  sandwiched_couples=sandwiched_couples
                )
        networks_data = [C_xys, C_yxs, T_xys, T_yxs]
        return networks_data

    network_windows_array = np.empty((n_frames,4,n_nodes,n_nodes))
    for frame_index in range(n_frames):
        data_window = get_data_window(frame_index,start_indices,window_size)
        network_windows_array[frame_index] = get_network_window(data_window)

    return network_windows_array


def plot_heatmaps(network_windows_array,
                node_labels,
                target_node,
                time,
                heatmap_max=0.5,
                window_size=100,
                step_size=5,
                plot_filename_prefix='heatmaps',
                return_figs=False):
    
    """
    Plot Cdiff and Tdiff heatmaps for all source nodes to the target node.

    network_windows_array: (n_frames, 4 coeffs, n_nodes, n_nodes)
    """

    n_time = len(time)
    n_frames,_,n_nodes,_ = network_windows_array.shape
    start_indices = np.arange(0, n_time - window_size, step_size)

    target_idx = node_labels.index(target_node)
    non_target_mask = np.array(node_labels) != target_node
    non_target_indices = np.where(non_target_mask)[0]


    diff_arrays = {
        # C_xy - C_yx
        'C_diff': (network_windows_array[:,0,target_idx,:] - network_windows_array[:,1,target_idx,:]).T[non_target_indices],
        # T_xy - T_yx
        'T_diff': (network_windows_array[:,2,target_idx,:] - network_windows_array[:,3,target_idx,:]).T[non_target_indices],
    }

    start_times = np.array([time[i] for i in start_indices])
    tick_step = 2
    t_min = np.ceil(start_times[0] / tick_step) * tick_step
    t_max = np.floor(start_times[-1] / tick_step) * tick_step
    clean_tick_values = np.arange(t_min, t_max + tick_step + 1, tick_step)
    time_tick_locs = np.interp(clean_tick_values, start_times, np.arange(len(start_times)))
    time_tick_labels = [int(t) for t in clean_tick_values]

    if return_figs:
        figs = []

    for coeff,heatmap_array in diff_arrays.items():

        heatmap_masked = np.ma.masked_less(heatmap_array,0)

        fig,ax = plt.subplots(figsize=(6,0.4*len(non_target_indices)))

        im = ax.imshow(heatmap_masked,
                       vmax=heatmap_max,
                       aspect='auto',
                       origin='upper',
                       cmap='viridis',
                       interpolation='none') 
        
        for k in range(len(non_target_indices)):
            ax.axhline(y=k + 0.5, color='white', linewidth=1.5)

        ax.set_xticks(time_tick_locs)
        ax.set_xticklabels(time_tick_labels)

        other_sensor_indices = np.arange(len(node_labels)-1)
        other_sensor_list = [node for node in node_labels if node != target_node]
        ax.set_yticks(other_sensor_indices)
        ax.set_yticklabels(other_sensor_list)

        ax.set_title(rf"{NET_DICT[coeff]}, $X=${target_node}")
        
        ax.set_xlabel("time (s)")
        cbar = plt.colorbar(im, ax=ax, fraction=0.06, pad=0.02)
        cbar.set_label(label=NET_DICT[coeff])

        fig.savefig(fname=f"{plot_filename_prefix}_{coeff}.svg", dpi=400)
            
        if return_figs:
            figs.append[fig]

        # plt.close()

    if return_figs:
        return figs


def animate_networks(network_windows_array,
                     mapping, target_nodes,
                     n_time,
                     window_size,
                     step_size,
                     width_scale = 5.0,
                     animation_filename = 'network_evolution.gif',
                     self_loops=False):

    FPS = 10 # Frames per second for the final GIF
    network_titles = ['C_xys Network', 'C_yxs Network', 'T_xys Network', 'T_yxs Network'] # Plot titles

    n_frames,_,n_nodes,_ = network_windows_array.shape
    start_indices = np.arange(0, n_time - window_size, step_size)

    # Prepare the figure and axes once
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes_flat = axes.flatten()

    # Pre-calculate the fixed circular layout (should not change over time)
    # We use a dummy graph just to get the node ordering for the layout.
    dummy_G = nx.DiGraph(np.zeros((n_nodes, n_nodes)))
    dummy_LG = nx.relabel_nodes(dummy_G, mapping)
    fixed_pos = nx.circular_layout(dummy_LG)

    # --- 1. Define a single function to draw a network on a specific axis ---
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
        ax.set_title(f"{title}\nTime Window: {t_start}-{t_start + window_size}", fontsize=12)

    # --- 2. The Update Function for the Animation ---
    def update(frame_index):
        """
        Function called by FuncAnimation for each frame.
        It calculates the networks for a new time window and updates the plots.
        """
        # Retreive the four networks, [C_xys, C_yxs, T_xys, T_yxs]
        networks_data = network_windows_array[frame_index]
        
        if not self_loops:
            for X_xys in networks_data:
                np.fill_diagonal(X_xys, 0)

        # Redraw all four subplots
        t_start = start_indices[frame_index]
        for i in range(4):
            ax = axes_flat[i]
            network_data = networks_data[i]
            title = network_titles[i]
            
            draw_single_network(ax, network_data, title, fixed_pos, t_start)
        
        # Return the updated artists (necessary for FuncAnimation)
        return axes_flat

    # --- 3. Run and Save the Animation ---
    print(f"Generating animation with {n_frames} frames...")

    anim = FuncAnimation(
        fig, 
        update, 
        frames=n_frames,
        blit=False,  # Set to False, as NetworkX often doesn't handle blitting well
        interval=1000/FPS # Delay between frames in ms
    )

    # Save the animation as a GIF
    writer = PillowWriter(fps=FPS)
    anim.save(animation_filename, writer=writer)

    plt.close(fig) # Close the figure to free up memory
    print(f"Animation saved as {animation_filename}")


