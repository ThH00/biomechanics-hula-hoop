import numpy as np
import scipy.signal as signal
from numpy.polynomial import Polynomial
from scipy.integrate import cumulative_simpson as integrate

def load_movella(file,
                 header_row=8,
                 lead_time=0):
    # Get the column labels
    with open(file, "r") as readfile:
        header_keys = readfile.readlines()[header_row-1].split(',')
    # Identify the column indices for each acceleration and Euler angle component
    time_index = header_keys.index('SampleTimeFine')
    x_accel = header_keys.index('Acc_X')
    y_accel = header_keys.index('Acc_Y')
    z_accel = header_keys.index('Acc_Z')
    x_angle = header_keys.index('Euler_X')
    y_angle = header_keys.index('Euler_Y')
    z_angle = header_keys.index('Euler_Z')
    x_omega = header_keys.index('Gyr_X')
    y_omega = header_keys.index('Gyr_Y')
    z_omega = header_keys.index('Gyr_Z\n')
    data = np.loadtxt(file,
                delimiter=",",
                skiprows=header_row, # Get all the rows after the header
                usecols=[time_index,                # time_column
                         x_accel,y_accel,z_accel,   # acceleration column (m/s^2)
                         x_angle,y_angle,z_angle,   # Euler angles column (deg)
                         x_omega,y_omega,z_omega]   # Angular velocities (deg/s)
                )
    # Change the units of time to seconds and start at zero
    data[:,0] = (data[:,0]-data[0,0])/1000000 
    # Subtract the lead time
    start_index = np.where(data[:,0]>lead_time)[0][0]
    data = data[start_index:]
    # Start at zero time again
    data[:,0] = data[:,0]-data[0,0]
    # Baseline correct by subtracting average acceleration of the first 100-200 samples
    data[:,1:4] = data[:,1:4] - np.mean(data[100:200,1:4],axis=0) 
    # Convert angle to radians
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
                 accel_z):
    
    veloc_x = detrend(time[1:], integrate(y=accel_x,x=time))
    veloc_y = detrend(time[1:], integrate(y=accel_y,x=time))
    veloc_z = detrend(time[1:], integrate(y=accel_z,x=time))

    displ_x = detrend(time[2:], integrate(y=veloc_x,x=time[1:]))
    displ_y = detrend(time[2:], integrate(y=veloc_y,x=time[1:]))
    displ_z = detrend(time[2:], integrate(y=veloc_z,x=time[1:]))

    return displ_x, displ_y, displ_z, veloc_x, veloc_y, veloc_z