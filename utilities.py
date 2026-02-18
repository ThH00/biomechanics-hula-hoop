""" 
Created by Theresa E. Honein and Chrystal Chern as part of the
accompanying code and data for the paper, submitted in 2026, to the
Proceedings of the Royal Society A, "The Biomechanics of Hula Hooping"
by C. Chern, T. E. Honein, and O. M. O'Reilly.

Licensed under the GPLv3. See LICENSE in the project root for license information.

February 20, 2026

"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq
import networkx as nx
from network_computation import compute_functional_network
from pathlib import Path


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
    'ax': r"$a_{x}$",
    'ay': r"$a_{y}$",
    'az': r"$a_{z}$",
    'Ax': r"$A_{x}$",
    'Ay': r"$A_{y}$",
    'Az': r"$A_{z}$",
    'phi': r"$\phi$",
    'theta': r"$\theta$",
    'psi': r"$\psi$",
    'phidot': r"$\dot{\phi}$",
    'thetadot': r"$\dot{\theta}$",
    'psidot': r"$\dot{\psi}$"
}

SENSOR_DICT = {
    'hoop': 'h',
    'femur': 'f',
    'tibia': 't',
    'metatarsal': 'm'
}

NET_DICT = {
    'C_xys': r"$C_{xy}$",
    'T_xys': r"$T_{xy}$",
    'C_diff': r'$C_{\mathrm{diff}}$',
    'T_diff': r'$T_{\mathrm{diff}}$',
}


def data_to_array(data_dict,
                quantities={
                    'femur':['wx','wy','wz'],
                    'tibia':['wx','wy','wz'],
                    'metatarsal':['wx','wy','wz'],
                    'hoop':['wxy','psidot'],
                },
                ntime=None):
    """
    Returns an array with shape = (ntime, nquantities)
    Returns an array with shape = (1894,  11)
    """
    data_array = [data_dict[s][q] for s,qset in quantities.items() for q in qset]
    if ntime is not None:
        return np.array(data_array).T[:ntime]
    else:
        return np.array(data_array).T


def get_steady_hooping_interval(psi, dt=1.0, threshold=0.45, window_size=50):
    """
    The angle psi seems to be close to linear during steady hula hooping.
    We exploit this fact to determine the interval of steady hula hooping.
    
    Parameters:
    - psi: 1D numpy array of angle data
    - dt: time step between samples
    - threshold: maximum std deviation considered "steady"
    - window_size: size of moving window for std calculation
    
    Returns:
    - steady_intervals: list of (start_idx, end_idx)
    - averages: list of average psi values for each interval
    """

    # Unwrap and differentiate
    psi_unwrapped = np.unwrap(psi)
    dpsi_dt = np.gradient(psi_unwrapped, dt)

    # Time index
    t = np.arange(len(psi))

    # Moving window std deviation
    stds = np.array([np.std(dpsi_dt[i:i+window_size]) for i in range(len(psi) - window_size)])

    # Indices where std < threshold
    steady_indices = np.where(stds < threshold)[0]

    # Merge consecutive indices into intervals
    groups = []
    for k, g in groupby(enumerate(steady_indices), lambda ix: ix[0] - ix[1]):
        group = list(map(itemgetter(1), g))
        if len(group) >= window_size:
            groups.append((group[0], group[-1]+window_size))

    # Compute averages in each interval
    averages = [np.mean(dpsi_dt[start:end]) for start, end in groups]

    # Plot
    plt.figure(figsize=(10, 3))
    plt.plot(t, psi, label=r"$\psi$")
    plt.plot(t, dpsi_dt, label=r"$d\psi/dt$")
    for i, (start, end) in enumerate(groups):
        plt.axvspan(t[start], t[end], color='green', alpha=0.3, label='steady' if i == 0 else "")
        plt.text((t[start]+t[end])/2, psi[start], f'{averages[i]:.2f}', color='black', fontsize=12, ha='center', va='bottom')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return groups, averages


def fourier_spectrum(series, step):
    """
    Fourier amplitude spectrum of a signal, as a function of frequency.

    :param series:      time series.
    :type series:       1D array
    :param step:        timestep.
    :type step:         float

    :return:            (frequencies, amplitudes)
    :rtype:             tuple of arrays.
    """
    if series.ndim != 1:
        raise ValueError("series must be a 1D array.")
    N = len(series)
    frequencies = fftfreq(N,step)[1:N//2]
    amplitudes = 2.0/N*np.abs(fft(series)[1:N//2])

    return frequencies, amplitudes

def plot_FFT(freqencies,amplitudes,
             label=None,
             ax=None,title=True,subtitle=None,
             legend=False,
             alpha=1,
             xlim=None,
             color=None):
    if ax is None:
        fig,ax = plt.subplots(figsize=(4,2))
    ax.plot(freqencies,amplitudes,label=label,alpha=alpha,color=color)
    ax.set_xlim(xlim)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("Fourier Amplitude")
    if title:
        ax.set_title(f"FFT, {subtitle}")
    if legend:
        ax.legend(loc='upper left',bbox_to_anchor=(1.01, 1))



def plot_time_histories(sensor_labels,data_dict,time,title,y_limits=None,active_slice=None,one_per=False):
    sensors_to_plot = sensor_labels.keys()
    quantities = list(data_dict.values())[0].keys()
    if one_per:
        sensor_titles = [f"{SYMDICT[q]}" for q in quantities]
        
    else:
        sensor_titles = list(sensor_labels.values())
    
    colors = [
        'rgba(255, 0, 0, 0.7)',    # Red
        'rgba(0, 128, 0, 0.7)',    # Green
        'rgba(0, 0, 255, 0.7)',    # Blue
    ]

    # Keep track of which legend items have been shown
    legend_shown = [False, False, False]
    
    if one_per:
        ncols = len(quantities)
    else:
        ncols = 1

    fig = make_subplots(rows=len(sensors_to_plot), cols=ncols, subplot_titles=sensor_titles,
                        shared_xaxes=True, x_title="Time (s)",
                        shared_yaxes=True,
                        vertical_spacing=0.08)
    if active_slice is None:
        active_slice = slice(0,-1)
    # Loop through each sensor and subplot
    for i, sensor in enumerate(sensors_to_plot, start=1):
        for j, q in enumerate(quantities):
            fig.add_trace(go.Scatter(
                x=time[active_slice], 
                y=data_dict[sensor][q][active_slice], 
                mode='lines', 
                # name=f'{q.capitalize()}',
                # name=q,
                name = SYMDICT[q],
                # legendgroup=j,
                line=dict(color=colors[j]),
                showlegend=not one_per and not legend_shown[j]
            ), row=i, col=j+1 if one_per else 1)
            # Mark this legend item as shown after the first trace
            legend_shown[j] = True

        if y_limits is not None:
            # set y-axis range for this subplot
            fig.update_yaxes(range=y_limits[sensor], row=i, col=1)

    # Decrease margins and add a title
    fig.update_layout(
        height=150*len(sensors_to_plot),
        width=800,
        title_text=title,
        margin=dict(l=80, r=20, t=80, b=60),
    )
    fig.update_annotations(font_size=18)
    
    for annotation in fig.layout.annotations:
        if annotation.text in SYMDICT.values():
            annotation.update(
                y=1.05, font_size=22)

    if one_per:
        # Add y axes labels
        for i, sensor in enumerate(sensors_to_plot, start=1):
            fig.update_yaxes(title_text=f"{sensor_labels[sensor]}<br>(rad/s)", row=i, col=1)

    return fig

def get_top_frequencies(signal, fs, top_n=3):
    """Returns the top N FFT frequencies and magnitudes."""
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    mags = np.abs(fft_vals)

    if len(mags) > 1:
        # Find the top_n largest magnitudes, excluding the DC component (index 0)
        idx = np.argsort(mags[1:])[-top_n:] + 1
        idx = idx[::-1]  # descending order
        return freqs[idx], mags[idx]
    else:
        return np.array([]), np.array([])

def plot_top_frequencies(data_dict, data_axes, sampling_rate, sensor_colors, axis_markers):

    plt.figure(figsize=(12, 6))
    plt.title("Top 3 FFT Frequencies for Each Sensor Signal", fontsize=14)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Magnitude (log scale)", fontsize=12)
    plt.yscale('log')

    for sensor_name, axes_data in data_dict.items():
        color = sensor_colors.get(sensor_name, "black")

        for axis in data_axes:
            signal = axes_data[axis]
            freqs, mags = get_top_frequencies(signal, sampling_rate)

            plt.plot(
                freqs,
                mags,
                marker=axis_markers[axis],
                linestyle='',
                markersize=10,
                color=color,
                markeredgecolor='DarkSlateGrey',
                markeredgewidth=1,
                label=f"{sensor_name} - {axis}"
            )

    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(title="Sensor and Axis", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    return plt.gcf()

def plot_PCA_modes_by_segment(eigenvectors, quantities, n_modes=6):
    sensors = []
    mapping = [[] for _ in quantities]

    for i,(sensor,qset) in enumerate(quantities.items()):
        sensors.append(sensor)
        for q in qset:
            mapping[i].append(q)
    
    n_rows = n_modes
    n_sensors = len(sensors)
    n_cols = n_sensors

    fig,ax = plt.subplots(n_rows, n_cols,
                          gridspec_kw={'width_ratios': [3,3,3,1.6]},
                          figsize=(6.3, n_modes),
                          sharex='col',
                          sharey=True, 
                          constrained_layout=True
                          )

    col_idx = 0
    for j,s in enumerate(sensors): # Each column is a sensor with nq quantities
        nq = len(mapping[j])
        for i in range(n_rows): # Each row is a mode with n_senors sensors
            ax[i,j].plot(eigenvectors[i, col_idx:col_idx+nq], '.', markersize=12)
            ax[i,j].axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax[i,j].set_ylim(-1, 1)
            ax[i,0].set_ylabel(rf"$v_{{{i+1}}}$")
        col_idx += nq
        ax[-1,j].set_xticks(np.arange(nq))
        ax[-1,j].set_xticklabels(mapping[j],rotation=45)
        ax[-1,j].set_xlabel(s)
        ax[-1,-1].set_xlim(-0.1,1.11)
    
    fig.suptitle("Principal Component Analysis")

    return fig

def plot_PCA_variance_ratios(explained_variance_ratio):
    plt.figure(figsize=(6,4))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, color='skyblue', edgecolor='k')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f"Explained Variance Ratio of Principal Components")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return plt.gcf()

def plot_PCA_phase_portait(X_pca):
    plt.figure(figsize=(6,4), constrained_layout=True)
    plt.plot(X_pca[:,0],X_pca[:,1],'.')
    plt.title(f"Phase Portrait, First Two Principal Components")
    plt.xlabel(r"$\xi_{1}$")
    plt.ylabel(r"$\xi_{2}$")
    plt.grid(True)
    return plt.gcf()

def plot_PCA_FFT(X_pca,dt,n_modes=None,xlim=None,colors=None):
    if n_modes is None:
        n_modes = X_pca.shape[1]
    fig,ax = plt.subplots(figsize=(6,3), constrained_layout=True)
    if xlim is None:
        xlim = (0,5)
    for i in range(n_modes):
        freq,amp = fourier_spectrum(X_pca[:,i],step=dt)
        if colors is not None:
            plot_FFT(freq,amp,label=rf"$\xi_{{{i+1}}}$",ax=ax,title=False,legend=True,alpha=1,xlim=xlim,color=colors[i])
        else:
            plot_FFT(freq,amp,label=rf"$\xi_{{{i+1}}}$",ax=ax,title=False,legend=True,alpha=1,xlim=xlim)
    fig.suptitle(f"FFT")
    return fig



def data_to_array_by_quantity(data_dict,
                              quantities={
                                    'OR':[['time','wxy']],
                                    'IB':[['time','wx'], ['time','wy'], ['time','wz']],
                                    'IT':[['time','wx'], ['time','wy'], ['time','wz']],
                                    'IL':[['time','wx'], ['time','wy'], ['time','wz']],
                                },
                              ntime=None):
    """
    Returns an array with shape.      = (ntime, nquantities, ncomponents per quantity)
    For the default quantities, shape = (1894,  10,          2)
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


def run_network(data_dict,title,**config):
    quantities = config['quantities']
    target_node = config['target_node']
    verbose = config['verbose']
    save_arrays = config.get('save_arrays',False)
    out_dir = config.get('out_dir',Path('plots'))

    data = data_to_array_by_quantity(data_dict,
                            quantities=quantities,
                            ntime=config.get('ntime',None))
    if verbose:
        print(f"{title}: {data.shape=}")

    
    network = compute_functional_network(data,
                                    config.get('rr',(0.03,0.03,0.02)),
                                    n=config.get('n',np.shape(data)[1]),
                                    sandwiched_couples=config.get('sandwiched_couples',False),
                                    savez=config.get('savez',True),
                                    verbose=verbose
                                    )
    
    if save_arrays: # Save data and network arrays
        np.save(out_dir/f"{title}.npy", data)
        network_quantity_names = ['G', 'G_', 'common_G', 'T_diff', 'C_diff', 'C_xys', 'C_yxs', 'T_xys', 'T_yxs']
        network_quantities = {q: network[i] for i,q in enumerate(network_quantity_names)}
        for coeff in ['C_xys','T_xys','C_yxs','T_yxs']:
            np.save(out_dir/f"{title}_{coeff}.npy", network_quantities[coeff])
        if verbose:
            print(f"Saved data array as {out_dir}/{title}_data.npy")
            print(f"Saved network quantities as: \n"
                  f"{out_dir}/{title}_C_xys.npy, "
                  f"{out_dir}/{title}_T_xys.npy, "
                  f"{out_dir}/{title}_C_yxs.npy, "
                  f"{out_dir}/{title}_T_yxs.npy"
                  )
    
    mapping = {}
    idx = 0
    for sensor,qsets in quantities.items():
        for qset in qsets:
            mapping[idx] = f"{SYMDICT[qset[-1]]},{sensor}"
            idx += 1

    return network, mapping, target_node


def plot_network(coeff_xys,
                 mapping,
                 target_nodes,
                 width_scale=5.0,
                 self_loops=False,
                 title=None,
                 diffs=False,
                 draw_to_target_edges=True,
                 draw_from_target_edges=True,
                 draw_no_target_edges=True,
                 verbose=False):
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


    # Draw the Network
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
                      rr=(0.03,0.03,0.02),
                      C_threshold=0.02,
                      T_threshold=0.02,
                      window_size=100,
                      step_size=5,
                      sandwiched_couples=False,
                      verbose=False,
                      savez=False
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
        G, G_, common_G, T_diff, C_diff, C_xys, C_yxs, T_xys, T_yxs = compute_functional_network(
            data_window, rr=rr, C_threshold=C_threshold, T_threshold=T_threshold,
            n=n,  sandwiched_couples=sandwiched_couples, verbose=verbose, savez=savez,
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

        fig,ax = plt.subplots(figsize=(6,0.4*len(non_target_indices)), constrained_layout=True)

        im = ax.imshow(heatmap_masked,
                       vmax=heatmap_max,
                       aspect='auto',
                       origin='upper',
                       cmap='viridis',
                       interpolation='none',
                       rasterized=True
                       ) 
        
        for k in range(len(non_target_indices)):
            ax.axhline(y=k + 0.5, color='white', linewidth=1.5)

        ax.set_xticks(time_tick_locs)
        ax.set_xticklabels(time_tick_labels)

        other_sensor_indices = np.arange(len(node_labels)-1)
        other_sensor_list = [node for node in node_labels if node != target_node]
        ax.set_yticks(other_sensor_indices)
        ax.set_yticklabels(other_sensor_list)

        ax.set_title(rf"{NET_DICT[coeff]}, $X=${target_node}, Window={window_size}")
        
        ax.set_xlabel("time (s)")
        cbar = plt.colorbar(im, ax=ax, fraction=0.06, pad=0.02)
        cbar.set_label(label=NET_DICT[coeff])

        fig.savefig(fname=f"{plot_filename_prefix}_{coeff}_{window_size}.pdf", dpi=400)
            
        if return_figs:
            figs.append(fig)

    if return_figs:
        return figs




