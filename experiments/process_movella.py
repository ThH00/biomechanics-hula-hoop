"""
Data processing functions for acceleration and angular data
collected on Movella XSENS DOT IMU sensors.
Includes data loading, detrending, integrating, and filtering.
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from scipy.integrate import cumulative_simpson as integrate
from scipy.signal import butter, filtfilt


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

def lowpass_filter(signal, cutoff, fs=120, order=4, plot=False):
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

    if plot:
        fig,ax = plt.subplots(figsize=(10, 4),tight_layout=True)
        ax.plot(signal, label='Noisy signal', alpha=0.6)
        ax.plot(filtered, label='Filtered signal', linewidth=2)
        ax.legend()
        ax.set_ylabel('Amplitude')
        ax.set_title('Low-pass Filter to Remove High-Frequency Noise')
        ax.grid(True)
        plt.show()

    return filtered

