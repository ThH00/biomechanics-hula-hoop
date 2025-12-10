import inspect

import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from network_computation import compute_functional_network
from matplotlib.animation import FuncAnimation, PillowWriter
from contextlib import redirect_stdout
import io
from pyunicorn.timeseries.inter_system_recurrence_network import InterSystemRecurrenceNetwork

source_code_package = inspect.getsource(InterSystemRecurrenceNetwork)
print(source_code_package[:200]) # Print first 200 characters

from pathlib import Path
import os

SYMDICT = {
    'wx': r"$\omega_{x}$",
    'wy': r"$\omega_{y}$",
    'wz': r"$\omega_{z}$",
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
    'IL': 'c'
}

SENSOR_DICT_LONG = {
    'OR': 'hoop',
    'IB': 'femur',
    'IT': 'tibia',
    'IL': 'cunei'
}

PLAIN_DICT = {
        f"{n}{symbol}": f"{SENSOR_DICT_LONG[s]}_{name}"  # e.g. "femur_phi")
        for s,n in SENSOR_DICT.items()
        for name,symbol in SYMDICT.items()
    }

LONG_DICT = {
        f"{n}{symbol}": f"{SENSOR_DICT_LONG[s]} {symbol}"  # e.g. "femur $\phi$")
        for s,n in SENSOR_DICT.items()
        for name,symbol in SYMDICT.items()
    }

NET_DICT = {
    'C_xys': r"$C_{xy}$",
    'T_xys': r"$T_{xy}$",
}

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


def get_time_networks(data,
                      rr, C_threshold, T_threshold,
                      window_size=100,
                      step_size=5,
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

    def get_network_window(data_window,rr,C_threshold,T_threshold,n=n_nodes):
        """Get the networks data for the current window"""
        # Calculate the four networks
        with redirect_stdout(io.StringIO()): # suppress print statements
            G, G_, common_G, T_diff, C_diff, C_xys, C_yxs, T_xys, T_yxs = compute_functional_network(
                data_window, rr, C_threshold=C_threshold, T_threshold=T_threshold, n=n
            )
        networks_data = [C_xys, C_yxs, T_xys, T_yxs]
        return networks_data

    network_windows_array = np.empty((n_frames,4,n_nodes,n_nodes))
    for frame_index in range(n_frames):
        data_window = get_data_window(frame_index,start_indices,window_size)
        network_windows_array[frame_index] = get_network_window(data_window,rr,C_threshold,T_threshold)

    return network_windows_array


def plot_time_networks(network_windows_array,
                       mapping, target_nodes,
                       width_scale,
                       time,
                       window_size=100,
                       step_size=5,
                       plot_filename_prefix = 'network_time',
                       return_figs = False):

    n_time = len(time)
    n_frames,_,n_nodes,_ = network_windows_array.shape
    start_indices = np.arange(0, n_time - window_size, step_size)
        
    other_nodes = {s: name for s,name in mapping.items() if name not in target_nodes} # non-target nodes
    target_nodes = {s: name for s,name in mapping.items() if name in target_nodes} # target nodes
    arrow_dirs = ['to','from']
    net_types = ['C_xys','T_xys']

    # unique first letters of non-target node names
    sensor_set = {node_name[0] for node_name in other_nodes.values()}
    
    other_nodes_by_sensor = {
        sensor: {
            s: node_name
            for s,node_name in other_nodes.items()
            if node_name[0]==sensor
        }
        for sensor in sensor_set
    }
    
    if return_figs:
        figs = {
            arrow_dir: {
                sensor: {
                    net_type: None
                    for net_type in net_types
                }
                for sensor in sensor_set
            }
            for arrow_dir in arrow_dirs
        }

    nets = {
        arrow_dir: {
            target_s: {
                net_type: {
                    other_s: np.empty(n_frames)
                    for other_s in other_nodes.keys()
                }
                for net_type in net_types
            }
            for target_s in target_nodes.keys()
        }
        for arrow_dir in arrow_dirs
    }

    for target_s,target_name in target_nodes.items():
        for frame_idx in range(n_frames):
            C_xys,C_yxs,T_xys,T_yxs = network_windows_array[frame_idx]
            assert np.allclose(C_xys,C_yxs.T)
            assert np.allclose(T_xys,T_yxs.T)
            net_at_frame = {
                "C_xys": [width_scale*C_xys, width_scale*C_yxs],
                "T_xys": [width_scale*T_xys, width_scale*T_yxs]
            }
            for dir_idx,arrow_dir in enumerate(arrow_dirs):
                for net_type in net_types:
                    for other_s in other_nodes.keys():
                        # to: X_xys from non-target node to target node
                        # from: X_yxs from non-target node to target node
                        nets[arrow_dir][target_s][net_type][other_s][frame_idx] = net_at_frame[net_type][dir_idx][other_s][target_s]

    reverse_sensor_dict = {v:k for k,v in SENSOR_DICT.items()}
    for arrow_dir in arrow_dirs:
        for sensor in sensor_set:
            for net_type in net_types:

                # TODO: add network animation in left column
                # fig,ax = plt.subplots(nrows=3,ncols=2,
                #                     width_ratios=[1,2],
                #                     figsize=(10,8)
                #                 )
                fig,ax = plt.subplots(nrows=len(target_nodes),ncols=1,
                                      figsize=(10,2.5*len(target_nodes)),
                                      sharex=True,
                                      constrained_layout=True)
                for tn_idx,(target_s,target_name) in enumerate(target_nodes.items()):
                    if len(target_nodes) > 1:
                        tn_ax = ax[tn_idx]
                    else:
                        tn_ax = ax
                    for other_s,node_name in other_nodes_by_sensor[sensor].items():
                        tn_ax.plot([time[i] for i in start_indices],nets[arrow_dir][target_s][net_type][other_s], label=LONG_DICT[node_name])
                    tn_ax.set_title(rf"{NET_DICT[net_type]}, {SENSOR_DICT_LONG[reverse_sensor_dict[sensor]]} {arrow_dir} {LONG_DICT[target_name]}")
                    tn_ax.set_xlabel("time (s)")
                    tn_ax.legend()
                
                # e.g., network_time_plots/angular_velocities/to_hoop/
                filedir = Path(str(plot_filename_prefix))/f"{arrow_dir}_hoop/"
                if not filedir.exists():
                    os.makedirs(filedir)
                # e.g., femur_Cxys.png
                filename = filedir/f"{SENSOR_DICT_LONG[reverse_sensor_dict[sensor]]}_{net_type}.png"
                print(f"Saving {filename}")
                fig.savefig(fname=filename, dpi=400)
                
                if return_figs:
                    figs[arrow_dir][sensor][net_type] = fig

                plt.close()

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

