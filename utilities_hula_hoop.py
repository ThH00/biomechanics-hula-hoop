"""
Data analysis functions for hula hooping acceleration
and angular data records.
Includes steady interval detection, rotation extraction,
acceleration extraction in the fixed frame, 
initial position calculation, modal analysis,
PCA, and plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq


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