import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from network_computation import compute_functional_network
from matplotlib.animation import FuncAnimation, FFMpegWriter

def plot_rotation_networks(
    C_xys, C_yxs, T_xys, T_yxs,
    mapping,
    target_nodes=("h_psidot",),
    width_scale=50.0,
    savepath="network_visual_rotation_mix.pdf",
    figsize=(20, 20)
):
    """
    Build and plot C/T networks and their positive directional differences.

    Parameters
    ----------
    C_xys, C_yxs, T_xys, T_yxs : np.ndarray
        Adjacency matrices.
    mapping : dict
        Node index -> node name mapping.
    target_nodes : tuple or list
        Nodes of special interest (default: ("h_psidot",)).
    width_scale : float
        Scaling factor for edge widths.
    savepath : str
        Path to save the figure (PDF).
    figsize : tuple
        Figure size for matplotlib.
    """

    # --- Differences ---
    diff_C = C_xys - C_yxs
    diff_T = T_xys - T_yxs

    # Keep only positive values
    diff_Cpos = diff_C * (diff_C > 0)
    diff_Tpos = diff_T * (diff_T > 0)

    # Focus on edges from the first node only
    diff_Cpos[1:, :] = 0
    diff_Tpos[1:, :] = 0

    # Transpose so edges point *towards* h_psidot
    diff_Cpos = diff_Cpos.T
    diff_Tpos = diff_Tpos.T

    # Zero diagonals
    for arr in [C_xys, C_yxs, T_xys, T_yxs]:
        np.fill_diagonal(arr, 0)

    networks = [
        C_xys, C_yxs, T_xys, T_yxs,
        diff_Cpos, diff_Tpos
    ]
    network_titles = [
        "C_xys Network", "C_yxs Network",
        "T_xys Network", "T_yxs Network",
        "C_diff Network", "T_diff Network"
    ]

    # --- Plot setup ---
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes_flat = axes.flatten()

    for i, network_data in enumerate(networks):
        ax = axes_flat[i]

        G = nx.DiGraph(network_data)
        LG = nx.relabel_nodes(G, mapping)
        pos = nx.circular_layout(LG)

        # Edge categories
        special_edgelist = []   # from target
        special2_edgelist = []  # to target
        other_edgelist = []

        special_widths = []
        special2_widths = []
        other_widths = []

        for u, v, data_ in LG.edges(data=True):
            weight = data_.get("weight", 0)
            if weight <= 0:
                continue

            width = weight * width_scale
            is_from_target = u in target_nodes
            is_to_target = v in target_nodes

            if is_from_target and not is_to_target:
                special_edgelist.append((u, v))
                special_widths.append(width)
            elif is_to_target and not is_from_target:
                special2_edgelist.append((u, v))
                special2_widths.append(width)
            elif is_from_target and is_to_target:
                special_edgelist.append((u, v))
                special_widths.append(width)
            else:
                other_edgelist.append((u, v))
                other_widths.append(width)

        # Nodes
        nx.draw_networkx_nodes(
            LG, pos,
            node_size=700,
            node_color="lightgreen",
            edgecolors="black",
            ax=ax
        )
        nx.draw_networkx_labels(LG, pos, font_size=10, ax=ax)

        # Edges
        nx.draw_networkx_edges(
            LG, pos,
            edgelist=special_edgelist,
            width=special_widths,
            edge_color="red",
            alpha=0.7,
            arrowsize=15,
            connectionstyle="arc3,rad=0.1",
            ax=ax
        )

        nx.draw_networkx_edges(
            LG, pos,
            edgelist=special2_edgelist,
            width=special2_widths,
            edge_color="blue",
            alpha=0.7,
            arrowsize=15,
            connectionstyle="arc3,rad=0.1",
            ax=ax
        )

        nx.draw_networkx_edges(
            LG, pos,
            edgelist=other_edgelist,
            width=other_widths,
            edge_color="gray",
            alpha=0.1,
            arrowsize=10,
            connectionstyle="arc3,rad=0.1",
            ax=ax
        )

        ax.set_title(network_titles[i], fontsize=14)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(savepath, format="pdf", bbox_inches="tight")
    plt.show()

def save_network_arrays(
    folder,
    data=None,
    C_xys=None,
    C_yxs=None,
    T_xys=None,
    T_yxs=None
):
    """
    Save network-related arrays to disk for later comparison.

    Parameters
    ----------
    folder : str
        Path where arrays will be saved.
    data, C_xys, C_yxs, T_xys, T_yxs : np.ndarray or None
        Arrays to save. Any argument left as None is skipped.
    """

    os.makedirs(folder, exist_ok=True)

    if data is not None:
        np.save(os.path.join(folder, "data.npy"), data)
    if C_xys is not None:
        np.save(os.path.join(folder, "C_xys.npy"), C_xys)
    if C_yxs is not None:
        np.save(os.path.join(folder, "C_yxs.npy"), C_yxs)
    if T_xys is not None:
        np.save(os.path.join(folder, "T_xys.npy"), T_xys)
    if T_yxs is not None:
        np.save(os.path.join(folder, "T_yxs.npy"), T_yxs)

def draw_single_network(ax, network_data, title, pos, t_start, mapping, width_scale, target_nodes, WINDOW_SIZE):
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

def update(frame_index, data2, mapping, WINDOW_SIZE, start_indices, 
           axes_flat, network_titles, fixed_pos, width_scale, target_nodes,
           all_C_xys, all_C_yxs, all_T_xys, all_T_yxs):
    """
    Function called by FuncAnimation for each frame.
    It calculates the networks for a new time window and updates the plots.
    The accumulation lists (all_C_xys, etc.) are modified in place (by reference).
    """
    t_start = start_indices[frame_index]
    t_end = t_start + WINDOW_SIZE
    
    # 1. Slice the data for the current window
    data_window = data2[t_start:t_end, :]

    # 2. Calculate the four networks
    # This assumes 'compute_functional_network' is available in the environment
    # G, G_, common_G, C_xys, C_yxs, T_xys, T_yxs, RRx, RRy, RRxy = compute_functional_network_th(data_window,th=(0.03, 0.03, 0.02), n=N_NODES )
    _, _, _, _, _, C_xys, C_yxs, T_xys, T_yxs = compute_functional_network(data_window,rr=(0.03, 0.03, 0.02), n=10 )
    
    
    networks_data = [C_xys, C_yxs, T_xys, T_yxs]

    # --- APPEND MATRICES TO THE ACCUMULATION LISTS (Passed by Reference) ---
    all_C_xys.append(C_xys)
    all_C_yxs.append(C_yxs)
    all_T_xys.append(T_xys)
    all_T_yxs.append(T_yxs)

    # 3. Redraw all four subplots
    for i in range(4):
        ax = axes_flat[i]
        network_data = networks_data[i]
        title = network_titles[i]
        
        # Pass WINDOW_SIZE to the draw function
        draw_single_network(ax, network_data, title, fixed_pos, t_start, mapping, width_scale, target_nodes, WINDOW_SIZE)
    
    # Return the updated artists (necessary for FuncAnimation)
    return axes_flat

def get_sliding_window_animation(data2, mapping, network_titles, width_scale, target_nodes, WINDOW_SIZE = 20, STEP_SIZE = 5):
    '''
    get animation of sliding window saving the thickness of the network

        WINDOW_SIZE: Number of time points in the sliding window
        STEP_SIZE: How much the window shifts per frame (fewer steps = faster animation)
    '''
    
    # Number of nodes
    # Check shape compatibility
    if np.shape(data2)[1] % 2 != 0:
        raise ValueError("The number of columns in data2 must be even (2 * N_NODES).")
        
    N_NODES = int(np.shape(data2)[1]/2)


    # --- ANIMATION PARAMETERS ---
    TOTAL_TIME_POINTS = np.shape(data2)[0]
    FPS = 10             # Frames per second for the final GIF

    # Calculate the start indices for each frame
    start_indices = np.arange(0, TOTAL_TIME_POINTS - WINDOW_SIZE, STEP_SIZE)
    NUM_FRAMES = len(start_indices)

    if NUM_FRAMES == 0:
        print("Warning: Data is too short or window/step size is too large to generate any frames.")
        return None, None, None, None

    # Prepare the figure and axes once
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes_flat = axes.flatten()

    # Pre-calculate the fixed circular layout
    dummy_G = nx.DiGraph(np.zeros((N_NODES, N_NODES)))
    dummy_LG = nx.relabel_nodes(dummy_G, mapping)
    fixed_pos = nx.circular_layout(dummy_LG)

    # These lists will store the (N x N) matrices for each time window (frame).
    all_C_xys = []
    all_C_yxs = []
    all_T_xys = []
    all_T_yxs = []

    # --- 3. Run and Save the Animation ---

    print(f"Generating animation with {NUM_FRAMES} frames...")

    # Define the arguments (fargs) that need to be passed to the update function
    fargs_tuple = (
        data2, mapping, WINDOW_SIZE, start_indices, 
        axes_flat, network_titles, fixed_pos, width_scale, target_nodes,
        all_C_xys, all_C_yxs, all_T_xys, all_T_yxs # <-- The lists are included here!
    )
    
    anim = FuncAnimation(
        fig, 
        update, 
        frames=NUM_FRAMES,
        fargs=fargs_tuple, # <-- Pass the contextual arguments and the lists
        blit=False,  
        interval=1000/FPS 
    )
        
    # Set save path
    animation_filename = 'network_evolution.mp4'

    # Create writer
    writer = FFMpegWriter(fps=FPS)

    # Save animation as MP4
    anim.save(animation_filename, writer=writer)

    plt.close(fig) # Close the figure to free up memory
    print(f"Animation saved as {animation_filename}")

    # --- CONVERT LISTS TO NUMPY ARRAYS AND SAVE ---
    # The lists are now filled because the update function modified them in place.

    # Convert the lists of matrices into 3D NumPy arrays (Frames x Nodes x Nodes)
    C_xys_3D = np.array(all_C_xys)
    C_yxs_3D = np.array(all_C_yxs)
    T_xys_3D = np.array(all_T_xys)
    T_yxs_3D = np.array(all_T_yxs)

    # Save the 3D arrays to disk
    np.save('C_xys_matrices.npy', C_xys_3D)
    np.save('C_yxs_matrices.npy', C_yxs_3D)
    np.save('T_xys_matrices.npy', T_xys_3D)
    np.save('T_yxs_matrices.npy', T_yxs_3D)

    print("\n--- Matrix Data Saved ---")
    print(f"Shape of saved matrices ({NUM_FRAMES} frames x {N_NODES} nodes x {N_NODES} nodes): ({C_xys_3D.shape})")
    print(f"Matrices saved as:\n  - C_xys_matrices.npy\n  - C_yxs_matrices.npy\n  - T_xys_matrices.npy\n  - T_yxs_matrices.npy")

    return C_xys_3D, C_yxs_3D, T_xys_3D, T_yxs_3D

def plot_rotation_heatmaps(
    C_xys_3D, C_yxs_3D, T_xys_3D, T_yxs_3D,
    i_start=1,
    i_end=10,
    i_values=None,
    figsize=(15, 12),
    intervals=None,
    savepath="heatmaps_rotation_mix.pdf"
):
    """
    Plot heatmaps of C/T coupling terms and their directional differences
    for psidot_hoop against selected body angular velocities.

    Parameters
    ----------
    C_xys_3D, C_yxs_3D, T_xys_3D, T_yxs_3D : np.ndarray
        3D coupling tensors (time, i, j).
    i_start, i_end : int
        Slice indices along the 3rd dimension.
    i_values : array-like of str
        Labels for index i (length must match i_end - i_start).
    figsize : tuple
        Figure size.
    intervals : array-like or None
        X locations for vertical dashed lines.
    savepath : str
        Output PDF path.
    """

    if i_values is None:
        i_values = np.arange(i_start, i_end)
    i_labels = [str(i) for i in i_values]

    # --- Data extraction ---
    data_C0 = C_xys_3D[:, 0, i_start:i_end]
    data_T0 = T_xys_3D[:, 0, i_start:i_end]
    data_C2 = C_yxs_3D[:, 0, i_start:i_end]
    data_T2 = T_yxs_3D[:, 0, i_start:i_end]

    data_C4 = (
        C_xys_3D[:, 0, i_start:i_end]
        - C_xys_3D[:, i_start:i_end, 0]
    )
    data_T4 = (
        T_xys_3D[:, 0, i_start:i_end]
        - T_xys_3D[:, i_start:i_end, 0]
    )

    heatmap_info = [
        (data_C0, r"$C_{xy}$ (psidot_hoop)", r"$C_{xy}$"),
        (data_T0, r"$T_{xy}$ (psidot_hoop)", r"$T_{xy}$"),
        (data_C2, r"$C_{yx}$ (psidot_hoop)", r"$C_{yx}$"),
        (data_T2, r"$T_{yx}$ (psidot_hoop)", r"$T_{yx}$"),
        (data_C4, r"$diff_C$ (psidot_hoop)", r"$C_{xy}-C_{yx}$"),
        (data_T4, r"$diff_T$ (psidot_hoop)", r"$T_{xy}-T_{yx}$"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes = axes.flatten()

    if intervals is None:
        intervals = np.linspace(0, data_C0.shape[0] - 1, 22)

    for idx, (data, title, cbar_label) in enumerate(heatmap_info):
        ax = axes[idx]

        masked_data = np.ma.masked_less(data.T, 0)
        im = ax.imshow(
            masked_data,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="none",
        )

        ax.set_yticks(np.arange(len(i_values)))
        ax.set_yticklabels(i_labels)

        # Row separation lines
        for k in range(len(i_values) - 1):
            ax.axhline(y=k + 0.5, color="white", linewidth=1.5)

        # Vertical interval lines
        for x in intervals:
            ax.axvline(x=x, color="red", linestyle="--", linewidth=1)

        ax.set_xlabel("Slice Index")
        ax.set_ylabel(r"Index $i$")
        ax.set_title(title)

        plt.colorbar(im, ax=ax, label=cbar_label)

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.show()

def plot_positive_directional_differences(
    C_xys_3D,
    names,
    i_start=1,
    i_end=10,
    normalize_by_count=False,
):
    """
    Plot positive directional differences C_xys[:,0,i] - C_xys[:,i,0]
    and their average over indices i.

    Parameters
    ----------
    C_xys_3D : np.ndarray
        3D coupling tensor (time, i, j).
    names : list or dict
        Labels for each index i (must support names[i]).
    i_start, i_end : int
        Index range for i (i_start inclusive, i_end exclusive).
    normalize_by_count : bool
        If True, average only over positive contributions at each time.
        If False, divide by (i_end - i_start), matching your current code.
    """

    n_time = C_xys_3D.shape[0]

    diff_sum = np.zeros(n_time)
    pos_count = np.zeros(n_time)

    plt.figure()

    for i in range(i_start, i_end):
        diff = C_xys_3D[:, 0, i] - C_xys_3D[:, i, 0]
        pos = diff > 0

        diff_sum[pos] += diff[pos]
        pos_count[pos] += 1

        plt.plot(
            np.where(pos, diff, np.nan),
            label=names[i]
        )

    if normalize_by_count:
        avg_positive = diff_sum / pos_count
        avg_positive[pos_count == 0] = np.nan
    else:
        avg_positive = diff_sum / (i_end - i_start)
        avg_positive[pos_count == 0] = np.nan

    plt.plot(avg_positive, "k", linewidth=2, label="avg positive")
    plt.legend()

    plt.figure()
    plt.plot(pos_count, marker=".", linestyle="None")
    plt.xlabel("Time index")
    plt.ylabel("Positive count")

    plt.show()

    return avg_positive, pos_count    