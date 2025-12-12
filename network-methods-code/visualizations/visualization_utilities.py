import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Ellipse

def get_Ax(x,epsilon_x):
    '''
    Get recurrence network adjacency matrix (one time series)
        x: timeseries
        epsilon_x: distance threshold
    '''
    Nx = np.size(x)
    Ax = np.zeros((Nx,Nx))
    for i in range(Nx):
        for j in range(Nx):
            Ax[i,j] = np.heaviside(epsilon_x-np.abs(x[i]-x[j]),0)
    Ax = Ax-np.eye(Nx)
    rr = (np.sum(Ax)+np.shape(Ax)[0])/np.size(Ax)
    return Ax, rr

def get_Ax2(t,x,epsilon_x):
    '''
    Get recurrence network adjacency matrix (one time series)
        x: timeseries
        t: time array
        epsilon_x: distance threshold
    '''
    Nx = np.size(x)
    Ax = np.zeros((Nx,Nx))
    for i in range(Nx):
        for j in range(Nx):
            Ax[i,j] = np.heaviside(epsilon_x-np.linalg.norm([t[i]-t[j],x[i]-x[j]]),0)
    Ax = Ax-np.eye(Nx)
    rr = (np.sum(Ax)+np.shape(Ax)[0])/np.size(Ax)
    return Ax, rr


def plot_Ax(t, x, Ax, color, label, ax=None):
    """
    Helper function to plot a single time series and its internal connections.
    """
    if ax is None:
        ax = plt.gca()

    # Plot the time series
    ax.plot(t, x, '-o', color=color, label=label, alpha=0.7, linewidth=1)
    
    # Convert adjacency matrix to graph
    G = nx.from_numpy_array(Ax)
    
    # Draw edges (Auto-recurrence)
    # We use a simplified loop here. For very large datasets, LineCollection is faster.
    for i, j in G.edges():
        # Draw a curved or straight line between recurring points
        ax.plot([t[i], t[j]], [x[i], x[j]], '-', color=color, alpha=0.6, linewidth=1)

        ellipse = Ellipse(xy=((t[i]+t[j])/2,(x[i]+x[j])/2),       # center (x0, y0)
                    width=0.5,          # major axis length
                    height=0.5,         # minor axis length
                    angle=0,            # rotation in degrees
                    edgecolor=color,
                    facecolor='none',
                    alpha=0.4,
                    linewidth=0.5)

        ax.add_patch(ellipse)
    
    ax.set_ylabel(f"{label}(t)")
    ax.legend(loc='upper right')
    ax.grid(True)


def get_Axy(x,y,epsilon_x,epsilon_y,epsilon_xy):
    Nx = np.size(x)
    Ny = np.size(y)
    CRxy = np.zeros((Nx,Ny))
    #
    Ax, RRx = get_Ax(x,epsilon_x)
    print(f'RRx = {RRx}')
    Ay, RRy = get_Ax(y,epsilon_y)
    print(f'RRy = {RRy}')
    for i in range(Nx):
        for j in range(Ny):
            CRxy[i,j] = np.heaviside(epsilon_xy-np.abs(x[i]-y[j]),0)
    Axy = np.block([[Ax, CRxy],
                   [CRxy.T, Ay]])
    RRxy = np.sum(CRxy)/(Nx*Ny)
    print(f'RRxy = {RRxy}')
    return Axy, RRxy

def get_Axy2(t,x,y,epsilon_x,epsilon_y,epsilon_xy):
    Nx = np.size(x)
    Ny = np.size(y)
    CRxy = np.zeros((Nx,Ny))
    #
    Ax, RRx = get_Ax2(t,x,epsilon_x)
    # print(f'RRx = {RRx}')
    Ay, RRy = get_Ax2(t,y,epsilon_y)
    # print(f'RRy = {RRy}')
    for i in range(Nx):
        for j in range(Ny):
            CRxy[i,j] = np.heaviside(epsilon_xy-np.linalg.norm([x[i]-y[j],t[i]-t[j]]),0)
    Axy = np.block([[Ax, CRxy],
                   [CRxy.T, Ay]])
    RRxy = np.sum(CRxy)/(Nx*Ny)
    # print(f'RRxy = {RRxy}')

    return Axy, RRx, RRy, RRxy


def plot_Axy(t, x, y, Axy):
    '''
    Plotting connections on the time series in 3 subplots:
    1. X Auto-recurrence
    2. Y Auto-recurrence
    3. Cross-recurrence
    '''

    Nx = np.size(x)
    
    # Slice the matrix
    Ax = Axy[:Nx, :Nx]        # Auto-recurrence X
    Ay = Axy[Nx:, Nx:]        # Auto-recurrence Y
    CRxy = Axy[:Nx, Nx:]      # Cross-recurrence
    
    # Create 3 subplots, sharing the X axis (time)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    
    # --- Subplot 1: X Auto-recurrence ---
    plot_Ax(t, x, Ax, 'blue', 'x', ax=ax1)
    ax1.set_title("X Auto-recurrence")

    # --- Subplot 2: Y Auto-recurrence ---
    plot_Ax(t, y, Ay, 'green', 'y', ax=ax2)
    ax2.set_title("Y Auto-recurrence")

    # --- Subplot 3: Cross-recurrence (Original logic) ---
    ax3.plot(t, x, '-o', color='blue', label='x', alpha=0.6)
    ax3.plot(t, y, '-o', color='green', label='y', alpha=0.6)
    
    G_cross = nx.from_numpy_array(CRxy)
    
    # Draw edges connecting X to Y based on CRxy
    for i, j in G_cross.edges():
        # i is index in x, j is index in y
        ax3.plot([t[i], t[j]], [x[i], y[j]], 'r-', alpha=0.5, linewidth=1)
    
        ellipse = Ellipse(xy=((t[i]+t[j])/2,(x[i]+y[j])/2),       # center (x0, y0)
                    width=0.3,         # major axis length
                    height=0.3,        # minor axis length
                    angle=0,        # rotation in degrees
                    edgecolor='red',
                    facecolor='none',
                    alpha=0.5,
                    linewidth=1)

        ax3.add_patch(ellipse)
        
    ax3.set_ylabel("Amplitude")
    ax3.set_xlabel("Time")
    ax3.set_title("Cross-recurrence (x to y)")
    ax3.legend(loc='upper right')
    ax3.grid(True)

    plt.tight_layout()

    # Save as EPS
    fig.savefig("Axy_plot.eps", format="eps", dpi=300)
    
    plt.show()



def plot_Axy_frame(t, x, y, Axy, axes):
    '''
    Refactored to accept specific axes (ax1, ax2, ax3)
    axes: list or tuple of 3 matplotlib axis objects
    '''
    ax1, ax2, ax3 = axes
    
    Nx = np.size(x)
    
    # Slice the matrix
    Ax = Axy[:Nx, :Nx]        # Auto-recurrence X
    Ay = Axy[Nx:, Nx:]        # Auto-recurrence Y
    CRxy = Axy[:Nx, Nx:]      # Cross-recurrence
    
    # --- Subplot 1: X Auto-recurrence ---
    plot_Ax(t, x, Ax, 'blue', 'x', ax=ax1)
    ax1.set_title("X Auto-recurrence")
    ax1.grid(True)

    # --- Subplot 2: Y Auto-recurrence ---
    plot_Ax(t, y, Ay, 'green', 'y', ax=ax2)
    ax2.set_title("Y Auto-recurrence")
    ax2.grid(True)

    # --- Subplot 3: Cross-recurrence ---
    ax3.plot(t, x, '-o', color='blue', label='x', alpha=0.6)
    ax3.plot(t, y, '-o', color='green', label='y', alpha=0.6)
    
    G_cross = nx.from_numpy_array(CRxy)
    
    # Draw edges connecting X to Y based on CRxy
    for i, j in G_cross.edges():
        # i is index in x, j is index in y
        # Safety check to ensure indices are within bounds of the current window
        if i < len(t) and j < len(t):
            ax3.plot([t[i], t[j]], [x[i], y[j]], 'r-', alpha=0.5, linewidth=1)
        
    ax3.set_title("Cross-recurrence (x to y)")
    ax3.grid(True)
    # Note: We removed plt.show() and plt.tight_layout()

def get_Cv_XY(v,Nx,Ny,Axy):    
    """
    Calculates the centrality or clustering contribution C_v^XY for a given node v,
    based on the provided mathematical formula.

    The formula is:
    C_v^XY = (1 / (k_v^XY * (k_v^XY - 1))) * Sum_{p,q in V^Y} (A_vp * A_pq * A_qv)

    v is an index of x
    """

    Ax = Axy[:Nx, :Nx]
    Ay = Axy[Nx:, Nx:]
    CRxy = Axy[:Nx, Nx:]

    kv_XY = np.sum(CRxy[v,:])

    # 1. Calculate the Denominator
    denominator = kv_XY * (kv_XY - 1)

    # 2. Calculate the Summation Term
    numerator = 0.0
    for p in range(Ny):
        for q in range(Ny):
            numerator = numerator + CRxy[v,p]*Ay[p,q]*CRxy[q,v]

    # 3. Final Calculation
    if denominator == 0.0:
        C_v_XY = 0
    else:
        C_v_XY = numerator / denominator

    return C_v_XY


def get_C_XY(Nx,Ny,Axy):
    temp = np.zeros(Nx)
    for i in range(Nx):
        temp[i] = get_Cv_XY(i,Nx,Ny,Axy)
    return np.mean(temp)



# def plot_time_series_connections(x,Ax,color):
#     '''Plotting connections on the time series'''
#     # Plot time series
#     plt.figure(figsize=(10,4))
#     plt.plot(t, x, '-o', color=color, label='Time Series')

#     # Convert adjacency matrix to graph
#     G = nx.from_numpy_array(Ax)

#     # Draw edges as lines connecting recurring points
#     for i, j in G.edges():
#         plt.plot([t[i], t[j]], [x[i], x[j]], '-', color=color,alpha=0.5)

#     plt.xlabel("Time")
#     plt.ylabel("x(t)")
#     plt.title("Time Series with Recurrence Network Edges")
#     plt.grid(True)
#     plt.show()

#     return




# def plot_network(Ax):
#     '''
#     Default network plotting from networkx package
#         Ax: adjacency matrix (one time series)
#     '''
    
#     # Create graph from adjacency matrix
#     G = nx.from_numpy_array(Ax)

#     # Draw the network
#     nx.draw(G, with_labels=True, node_color='skyblue', node_size=800, edge_color='gray')
#     plt.show()

#     return