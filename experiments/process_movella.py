import numpy as np
import scipy.signal as signal
from numpy.polynomial import Polynomial

def load_movella(file,
                 header_row=8,
                 lead_time=0):
    # Get the column labels
    with open(file, "r") as readfile:
        header_keys = readfile.readlines()[header_row-1].split(',')
    # Identify the column indices for each acceleration component
    time_index = header_keys.index('SampleTimeFine')
    x_index = header_keys.index('Acc_X')
    y_index = header_keys.index('Acc_Y')
    z_index = header_keys.index('Acc_Z')
    x_angle = header_keys.index('Euler_X')
    y_angle = header_keys.index('Euler_Y')
    z_angle = header_keys.index('Euler_Z')
    data = np.loadtxt(file,
                delimiter=",",
                skiprows=header_row, # Get all the rows after the header
                usecols=[time_index,x_index,y_index,z_index] # Get only the columns with acceleration data
                )
    # Change the units of time to seconds and start at zero
    data[:,0] = (data[:,0]-data[0,0])/1000000 
    # Subtract the lead time
    start_index = np.where(data[:,0]>lead_time)[0][0]
    data = data[start_index:]
    # Start at zero time again
    data[:,0] = data[:,0]-data[0,0]
    # Baseline correct by subtracting average acceleration of the first 100-200 samples
    data[:,1:] = data[:,1:] - np.mean(data[100:200,1:],axis=0) 

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
        pmial = Polynomial.fit(time,data,deg=degree)
        return data - pmial(time)
