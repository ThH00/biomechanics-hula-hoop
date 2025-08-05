"""
Data analysis functions for hula hooping acceleration
and angular data records.
Includes steady interval detection, rotation extraction,
initial position calculation, and modal analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter
from scipy.signal import find_peaks
from mdof import modes, outid, modal, transform


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
    plt.plot(t, psi, label='psi')
    plt.plot(t, dpsi_dt, label='dpsi_dt')
    for i, (start, end) in enumerate(groups):
        plt.axvspan(t[start], t[end], color='green', alpha=0.3, label='steady' if i == 0 else "")
        plt.text((t[start]+t[end])/2, psi[start], f'{averages[i]:.2f}', color='black', fontsize=8, ha='center', va='bottom')
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
    extract the local basis (e1,e2,e3)
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

def offset_sensor(phi0,theta0,psi0,
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
            plt.plot(periods,amplitudes)
        return fundamental_p[0]
    elif method == 'psd':
        P,Phi = outid(signal, dt=1/fs)
        return P[0]
    elif method == 'srim':
        if inputs == None:
            inputs = np.random.normal(loc=0, scale=1, size=signal.size)
        P,Phi = modes(inputs=inputs, outputs=signal, dt=1/fs,
                      method='srim', order=2)
        return P[0]
    elif method == 'okid':
        if inputs == None:
            inputs = np.random.normal(loc=0, scale=1, size=signal.size)
        P,Phi = modes(inputs=inputs, outputs=signal, dt=1/fs,
                     method='okid-era', order=2)
        return P[0]
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