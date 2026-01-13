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
from scipy.signal import find_peaks
from mdof import modes, outid, modal, transform
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

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

def extract_rotation(phi, theta, psi,
                     E1=np.array([1,0,0]),
                     E2=np.array([0,1,0]),
                     E3=np.array([0,0,1])):
    """
    From Euler angles psi on E3, theta on e2', and phi on e1'',
    extract the local basis (e1,e2,e3) in terms of the fixed
    basis (E1,E2,E3)
    """
    # First rotation by an angle psi about E3
    e1p =  np.cos(psi)*E1 + np.sin(psi)*E2
    e2p = -np.sin(psi)*E1 + np.cos(psi)*E2
    e3p = E3
    # Second rotation by an angle theta about e2p
    e1pp = np.cos(theta)*e1p - np.sin(theta)*e3p
    e2pp = e2p
    e3pp = np.sin(theta)*e1p + np.cos(theta)*e3p
    # Third rotation by an angle phi about e1pp
    e1 = e1pp
    e2 =  np.cos(phi)*e2pp+np.sin(phi)*e3pp
    e3 = -np.sin(phi)*e2pp+np.cos(phi)*e3pp

    return e1, e2, e3

def get_fixed_frame_acceleration(ax, ay, az,
                                 psi, theta, phi,
                                 E1=np.array([1,0,0]),
                                 E2=np.array([0,1,0]),
                                 E3=np.array([0,0,1])):

    n = np.size(psi)
    a = np.zeros((3,n))

    for i in range(n):
        e1,e2,e3 = extract_rotation(phi[i],theta[i],psi[i],
                                    E1=E1,E2=E2,E3=E3)
        # acceleration in fixed frame
        a[:,i] = ax[i]*e1 + ay[i]*e2 + az[i]*e3
    # a = A1*E1+A2*E2+A3*E3
    Ax = a[0,:]
    Ay = a[1,:]
    Az = a[2,:]

    return Ax, Ay, Az

def get_euler_derivatives(phi,theta,psi,
                          wx,wy,wz):
    transformation = np.array([
        [np.zeros_like(phi), np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)],
        [np.zeros_like(phi), np.cos(phi),               -np.sin(phi)],
        [np.ones_like(phi),  np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)]
        ]).transpose(2,0,1)
    corotational_angular_velocities = np.array([[wx],[wy],[wz]]).transpose(2,0,1)
    psidot, thetadot, phidot = (transformation @ corotational_angular_velocities).transpose(1,2,0)
    return phidot.flatten(), thetadot.flatten(), psidot.flatten()

def offset_hoop_sensor(phi0,theta0,psi0,
                  dx0,dy0,dz0,
                  angle_along_hoop=141.5,
                  radius=83.0/100/2):
    """
    Offset the initial location of a sensor that is a certain
    angle offset from the reference sensor.
    phi0, theta0, psi0, dx0, dy0, and dz0 are of the initial
    angle and position of the reference sensor.
    """
    delta = np.deg2rad(angle_along_hoop)
    ex, ey, ez = extract_rotation(phi0, theta0, psi0)
    center_of_hoop = np.array([dx0,dy0,dz0]) - radius*ey
    ray = np.cos(delta)*ex + np.sin(delta)*ey
    initial_position = center_of_hoop + radius*ray
    return initial_position

def estimate_period(signal, method='psd',
                    fs=120.0, plot=False,
                    inputs=None):
    """
    Estimate the period of the signal. Available methods include:
    fft:      Fast Fourier Transform
    psd:      FFT of power spectral density
    srim:     Time domain system identification by System Realization by
              Information Matrix. Needs inputs; if None, uses Gaussian
              white noise as input.
    okid:     Time domain system identification by Observer Kalman 
              Identification Algorithm. Needs inputs; if None, uses
              Gaussian white noise as input.
    autocorr: Find the lag that maximizes autocorrelation.
    """

    method = method.lower()

    if method == 'fft':
        periods, amplitudes = transform.fourier_spectrum(signal, step=1/fs)
        fundamental_p, fundamental_a = modal.spectrum_modes(periods, amplitudes,
                                                            sorted_by='height')
        if plot:
            fig,ax = plt.subplots()
            ax.plot(periods,amplitudes)
            ax.set_xlim(0,2)
            ax.set_title('Fast Fourier Transform')
            ax.set_xlabel('Period (s)')
            ax.set_ylabel('Amplitude')
            plt.show()
        return fundamental_p[0]
    elif method == 'psd':
        periods, amplitudes = transform.power_spectrum(signal, step=1/fs)
        fundamental_p, fundamental_a = modal.spectrum_modes(periods, amplitudes,
                                                            sorted_by='height')
        if plot:
            fig,ax = plt.subplots()
            ax.plot(periods,amplitudes)
            ax.set_xlim(0,2)
            ax.set_title('Power Spectral Density')
            ax.set_xlabel('Period (s)')
            ax.set_ylabel('Amplitude')
            plt.show()
        return fundamental_p[0]
    elif method == 'srim':
        if inputs == None:
            inputs = np.random.normal(loc=0, scale=1, size=signal.size)
        P,Phi = modes(inputs=inputs, outputs=signal, dt=1/fs,
                      method='srim', order=2)
        if len(P)>0:
            return P[0]
        else:
            warnings.warn("No natural periods found by SRIM method.")
    elif method == 'okid':
        if inputs == None:
            inputs = np.random.normal(loc=0, scale=1, size=signal.size)
        P,Phi = modes(inputs=inputs, outputs=signal, dt=1/fs,
                     method='okid-era', order=2)
        if len(P)>0:
            return P[0]
        else:
            warnings.warn("No natural periods found by SRIM method.")
    elif method=='autocorr':
        period_samples,period_time = estimate_period_autocorr(signal,fs,plot)
        return period_time

def estimate_period_autocorr(signal, fs=120.0, plot=False):
    signal = signal - np.mean(signal)
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    # Find all peaks
    peaks, _ = find_peaks(autocorr)

    if len(peaks) < 1:
        raise ValueError("No peaks found in autocorrelation.")

    # First peak after lag = 0 is the period
    period_samples = peaks[0]
    period_time = period_samples / fs

    if plot:
        _, ax = plt.subplots()
        ax.plot(autocorr)
        ax.axvline(period_samples, color='r', linestyle='--', label=f'Period ≈ {period_samples} samples ({period_time:.2f} s)')
        ax.set_title("Autocorrelation")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.legend()
        plt.show()

    return period_samples, period_time


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

def plot_PCA_FFT(X_pca,dt,subtitle,n_modes,xlim,colors=None):
    fig,ax = plt.subplots(figsize=(6,3))
    for i in range(n_modes):
        freq,amp = fourier_spectrum(X_pca[:,i],step=dt)
        if colors is not None:
            plot_FFT(freq,amp,label=rf"$\xi_{{{i+1}}}$",ax=ax,title=False,legend=True,alpha=1,xlim=xlim,color=colors[i])
        else:
            plot_FFT(freq,amp,label=rf"$\xi_{{{i+1}}}$",ax=ax,title=False,legend=True,alpha=1,xlim=xlim)
    fig.suptitle(f"FFT, {subtitle}")


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

def perform_PCA(data_dict,active_slice,
                sensors_to_include,quantities_to_include,
                verbose=False):
    
    if verbose:
        print(f"Quantities: {[f"{s}:{q}" for s in sensors_to_include for q in quantities_to_include]}")

    # Step 1: Stack and transpose
    if active_slice is not None:
        X = np.vstack([data_dict[s][q][active_slice] for s in sensors_to_include for q in quantities_to_include]).T
    else:
        X = np.vstack([data_dict[s][q] for s in sensors_to_include for q in quantities_to_include]).T

    if verbose:
        print(f"shape(X): {np.shape(X)}")

    # Step 2: Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: Apply PCA
    pca = PCA(n_components=np.shape(X)[1])
    X_pca = pca.fit_transform(X_scaled)
    # Get the eigenvectors (principal directions)
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    explained_variance_ratio = pca.explained_variance_ratio_

    # After fitting PCA
    if verbose == 2:
        print(f"{X_pca=}")
    if verbose:
        print("Eigenvalues:")
        print(eigenvalues)
    if verbose == 2:
        print("Eigenvectors (Principal Directions):")
        print(eigenvectors)
    if verbose:
        print("Explained Variance Ratio:")
        print(explained_variance_ratio)

    return X_pca, eigenvalues, eigenvectors, explained_variance_ratio

def plot_PCA(X_pca,domain,active_slice,domain_label="Time (s)",subtitle=None,separate=False):
    n_components = X_pca.shape[1]
    if separate:
        n = X_pca.shape[1]
        fig, ax = plt.subplots(n,1,figsize=(8,n_components),constrained_layout=True)
        for i in range(n):
            if active_slice is not None:
                ax[i].plot(domain[active_slice],X_pca[:,i]) 
            else:
                ax[i].plot(domain,X_pca[:,i]) 
            ax[i].set_ylabel(rf"$\xi_{{{i+1}}}$")
        ax[-1].set_xlabel(domain_label)
        fig.suptitle(f"Principal Components Over {domain_label} (Scores)\n{subtitle}")
    else:
        plt.figure(figsize=(10, 4))
        if active_slice is not None:
            plt.plot(domain[active_slice], X_pca)
        else:
            plt.plot(domain, X_pca)
        plt.xlabel(domain_label)
        plt.title(f"Principal Components Over {domain_label} (Scores)\n{subtitle}")
        plt.legend([r"$\xi_{1}$", r"$\xi_{2}$", r"$\xi_{3}$"])
        plt.ylabel("Score")
        plt.grid(True)
        plt.show()

def plot_PCA_eigenvalues(eigenvalues,subtitle=None):
    plt.figure(figsize=(8,4))
    plt.bar(range(1, len(eigenvalues) + 1), eigenvalues, color='skyblue', edgecolor='k')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.title(f"Eigenvalues of Principal Components\n{subtitle}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_PCA_variance_ratios(explained_variance_ratio,subtitle=None):
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, color='skyblue', edgecolor='k')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f"Explained Variance Ratio of Principal Components\n{subtitle}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_Scree(explained_variance_ratio,subtitle=None):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'o-', color='blue')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f"Scree Plot\n{subtitle}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_Cumul_Scree(explained_variance_ratio,subtitle=None):
    cumulative_variance = explained_variance_ratio.cumsum()

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-', color='green')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f"Cumulative Scree Plot\n{subtitle}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_PCA_phase_portait(X_pca,subtitle=None):
    plt.figure(figsize=(10, 4))
    plt.plot(X_pca[:,0],X_pca[:,1],'.')
    plt.title(f"Phase Portrait, First Two Principal Components\n{subtitle}")
    plt.xlabel(r"$\xi_{1}$")
    plt.ylabel(r"$\xi_{2}$")
    plt.grid(True)
    plt.show()

def plot_PCA_modes(eigenvectors,sensors_to_include,quantities_to_include,sensor_labels,n_modes):
    n_sensors_total = eigenvectors.shape[1]
    n_sensors = len(sensors_to_include)
    n_quantities = len(quantities_to_include)
    n_rows = n_modes
    n_cols = n_sensors
    x_vals = np.arange(n_quantities)

    _,ax = plt.subplots(n_rows, n_cols, figsize=(0.6*n_sensors_total, n_modes),
                           constrained_layout=True, sharex=True, sharey=True)

    for i in range(n_rows): # Each row is a mode with n_senors sensors
        for j in range(n_cols): # Each column is a sensor with n_quantities quantities
            ax[i,j].plot(eigenvectors[i, n_quantities*j:n_quantities*(j+1)], '.', markersize=12)
            # ax[i,j].stem(x_vals, eigenvectors[i, n_quantities*j:n_quantities*(j+1)])
            ax[i,j].axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax[i,j].set_xticks(x_vals)
            ax[i,j].set_xticklabels(quantities_to_include,rotation=45)
            ax[-1,j].set_xlabel(sensor_labels[sensors_to_include[j]])
            ax[i,j].set_ylim(-1, 1)
        ax[i,0].set_ylabel(rf"$\xi_{{{i+1}}}$")

        plt.savefig('PCA_modes.eps', format='eps')

def plot_PCA_modes_3D():
    return


from stl import mesh

def center_and_scale_stl(stl_mesh, desired_height):
    """
    Centers an STL mesh at the origin and scales it to a specific height.

    Args:
        input_file_path (str): The path to the input STL file.
        output_file_path (str): The path to save the modified STL file.
        desired_height (float): The target height for the scaled model.
    """

    # Calculate the bounding box of the mesh
    min_coords = stl_mesh.vectors.min(axis=0).min(axis=0)
    max_coords = stl_mesh.vectors.max(axis=0).max(axis=0)

    # Calculate dimensions and center
    width = max_coords[0] - min_coords[0]
    depth = max_coords[1] - min_coords[1]
    height = max_coords[2] - min_coords[2]
    center_offset = (min_coords + max_coords) / 2

    # Step 1: Center the mesh at the origin
    # Subtract the center of the bounding box from all mesh points
    stl_mesh.vectors -= center_offset

    # Step 2: Scale the mesh to the desired height
    # Calculate the uniform scaling factor
    scale_factor = desired_height / height
    
    # Apply the scaling to all mesh points
    stl_mesh.vectors *= scale_factor

    return stl_mesh
