import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from scipy.integrate import cumulative_simpson as integrate
from itertools import groupby
from operator import itemgetter
from scipy.signal import butter, filtfilt, find_peaks


def load_movella(file,
                 header_row=8,
                 lead_time=0,
                 end_time=0):
    # Get the column labels
    with open(file, "r") as readfile:
        header_keys = readfile.readlines()[header_row-1].split(',')
    # Identify the column indices for each acceleration and Euler angle component
    time_index = header_keys.index('SampleTimeFine')
    x_accel = header_keys.index('Acc_X')
    y_accel = header_keys.index('Acc_Y')
    z_accel = header_keys.index('Acc_Z')
    w_angle = header_keys.index('Quat_W') if 'Quat_W' in header_keys else None
    x_angle = header_keys.index('Euler_X') if 'Euler_X' in header_keys else header_keys.index('Quat_X')
    y_angle = header_keys.index('Euler_Y') if 'Euler_Y' in header_keys else header_keys.index('Quat_Y')
    z_angle = header_keys.index('Euler_Z') if 'Euler_X' in header_keys else header_keys.index('Quat_Z')
    x_omega = header_keys.index('Gyr_X')
    y_omega = header_keys.index('Gyr_Y')
    z_omega = header_keys.index('Gyr_Z\n') if 'Gyr_Z\n' in header_keys else header_keys.index('Gyr_Z')
    if w_angle is None:
        usecols=[time_index,                # time_column
                 x_accel,y_accel,z_accel,   # acceleration column (m/s^2)
                 x_angle,y_angle,z_angle,   # Euler angles column (deg)
                 x_omega,y_omega,z_omega]   # Angular velocities (deg/s)
    else:
        usecols=[time_index,                         # time_column
                 x_accel,y_accel,z_accel,            # acceleration column (m/s^2)
                 w_angle, x_angle,y_angle,z_angle,   # Quaternions column (deg)
                 x_omega,y_omega,z_omega]            # Angular velocities (deg/s)
    data = np.loadtxt(file,
                delimiter=",",
                skiprows=header_row, # Get all the rows after the header
                usecols=usecols
                )
    # Change the units of time to seconds and start at zero
    data[:,0] = (data[:,0]-data[0,0])/1000000 
    # Baseline correct by subtracting average acceleration of the first 100-200 samples
    data[:,1:4] = data[:,1:4] - np.mean(data[100:200,1:4],axis=0) 
    # Subtract the lead time
    start_index = np.where(data[:,0]>lead_time)[0][0]
    end_index = np.where(data[:,0]<data[-1,0]-end_time)[0][-1]
    data = data[start_index:end_index]
    # Start at zero time again
    data[:,0] = data[:,0]-data[0,0]
    # Convert angle to radians and angular velocities
    data[:,4:] = data[:,4:]*np.pi/180

    return data

def detrend(time,
            data,
            backend='polynomial',
            degree=6,    # polynomial backend only
            fit='linear' # scipy backend only
            ):
    if backend=='scipy':
        return signal.detrend(data,type=fit)
    elif backend=='polynomial':
        pnom = Polynomial.fit(time,data,deg=degree)
        return data - pnom(time)
    
def get_position(time,
                 accel_x,
                 accel_y,
                 accel_z,
                 detrended=True,
                 degree=6):
    
    veloc_x = integrate(y=accel_x,x=time)
    veloc_y = integrate(y=accel_y,x=time)
    veloc_z = integrate(y=accel_z,x=time)
    
    if detrended:
        veloc_x = detrend(time[1:], veloc_x, degree=degree)
        veloc_y = detrend(time[1:], veloc_y, degree=degree)
        veloc_z = detrend(time[1:], veloc_z, degree=degree)

    displ_x = integrate(y=veloc_x,x=time[1:])
    displ_y = integrate(y=veloc_y,x=time[1:])
    displ_z = integrate(y=veloc_z,x=time[1:])

    if detrended:
        displ_x = detrend(time[2:], displ_x, degree=degree)
        displ_y = detrend(time[2:], displ_y, degree=degree)
        displ_z = detrend(time[2:], displ_z, degree=degree)

    return displ_x, displ_y, displ_z, veloc_x, veloc_y, veloc_z

def get_steady_hooping_interval(psi, dt=1.0, threshold=0.45, window_size=50):
    '''
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
    '''

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

def lowpass_filter(signal, cutoff, fs=120, order=4):
    """
    Apply a Butterworth low-pass filter to remove high-frequency noise.

    Parameters:
    - signal: input 1D signal
    - cutoff: desired cutoff frequency (Hz)
    - fs: sampling rate (Hz)
    - order: order of the filter

    Returns:
    - filtered signal
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, signal)

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(signal, label='Noisy signal', alpha=0.6)
    plt.plot(filtered, label='Filtered signal', linewidth=2)
    plt.legend()
    plt.ylabel('Amplitude')
    plt.title('Low-pass Filter to Remove High-Frequency Noise')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return filtered

# @Chrystal. Please check this.
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
        fig, ax = plt.subplots()
        plt.plot(autocorr)
        plt.axvline(period_samples, color='r', linestyle='--', label=f'Period ≈ {period_samples}')
        plt.title("Autocorrelation")
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.legend()
        plt.show()

    return period_samples, period_time