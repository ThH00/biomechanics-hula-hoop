# -*- coding: utf-8 -*-
""" main file for functional network computation

Part of the accompanying code for the paper "Exploring localization in nonlinear oscillator systems through 
network-based predictions" by C. Geier and N. Hoffmann published in Chaos 35 
(5) 2025 doi: 10.1063/5.0265366 . Available at https://arxiv.org/abs/2407.05497

Compute functional network from time series data.

To reproduce the results in the paper, run the code with settings:

1. Homogeneous system, initial conditions [0, 0.1], no added noise, 10s time
series length:
    # data directory
    data_directory_main = 'data/homogeneous_ic1'

    # define storage directory for the results
    result_directory_main = 'results/funcnet_homogeneous_ic1_10s_no_noise'
    os.mkdir(result_directory_main)

    network_computation(data_directory_main,
                        result_directory_main,
                        time_series_length=10,
                        noise_level=0)

2. Homogeneous system, initial conditions [0, 0.01], no added noise, 10s time
series length:
    # data directory
    data_directory_main = 'data/homogeneous_ic2'

    # define storage directory for the results
    result_directory_main = 'results/funcnet_homogeneous_ic2_10s_no_noise'
    os.mkdir(result_directory_main)

    network_computation(data_directory_main,
                        result_directory_main,
                        time_series_length=10,
                        noise_level=0)

3. Heterogeneous system, initial conditions [0, 0.1], no added noise, 10s time
series length:
    # data directory
    data_directory_main = 'data/heterogeneous_ic1'

    # define storage directory for the results
    result_directory_main = 'results/funcnet_heterogeneous_ic1_10s_no_noise'
    os.mkdir(result_directory_main)

    network_computation(data_directory_main,
                        result_directory_main,
                        time_series_length=10,
                        noise_level=0)

4. Homogeneous system, initial conditions [0, 0.1], with added noise, 10s time
series length:
    # data directory
    data_directory_main = 'data/homogeneous_ic1'

    # define storage directory for the results
    result_directory_main = 'results/funcnet_homogeneous_ic1_10s_added_noise'
    os.mkdir(result_directory_main)

    network_computation(data_directory_main,
                        result_directory_main,
                        time_series_length=10,
                        noise_level=1)

5. Homogeneous system, initial conditions [0, 0.1], no added noise, 25s time
series length:
    # data directory
    data_directory_main = 'data/homogeneous_ic1'

    # define storage directory for the results
    result_directory_main = 'results/funcnet_homogeneous_ic1_25s_no_noise'
    os.mkdir(result_directory_main)

    network_computation(data_directory_main,
                        result_directory_main,
                        time_series_length=25,
                        noise_level=1)


Copyright (c) Charlotte Geier
Hamburg University of Technology, Dynamics Group
www.tuhh.de/dyn
charlotte.geier@tuhh.de

Licensed under the GPLv3. See LICENSE in the project root for license information.

30.06.2024
"""


import networkx as nx
from pyunicorn.timeseries.inter_system_recurrence_network import InterSystemRecurrenceNetwork
import os
import numpy as np
from utils.utils import common_elements



def compute_functional_network_th(sol, th, **kwargs):
    """
    For a given time series sol, compute the inter-system-recurrence-network
    based on cross-clustering and cross-transitivity.
    - Iterative process: for each combi of two dofs:
        - generate isrn
        - compute cross-clustering coefficients, cross-transitivities and store in matrices C_xys, C_yxs and T_xys, T_yxs

    :param sol: time series data in [n_timesteps, 2*n], where n number of oscillators and sol has both displacements and velocities
    :param th: distance threshold to use for the computations. in th = (epsilon_x, epsilon_y, epsilon_xy). values related to noise
    :param kwargs: n: number of variables from the time series to use, if not the first half of variables in the
    time series
    :return: G based on C_diff, G_ based on T_diff and common_G which only contains common edges
    """
    #

    velocities_only = kwargs.get('velocities_only', False)
    sandwiched_couples = kwargs.get('sandwiched_couples', True)
    metric = kwargs.get('metric', 'euclidean')
    savez = kwargs.get('savez', True)

    # get number of variables from each time series to use
    # default = use first half, i.e. the positional ones
    n = kwargs.get('n', int(np.shape(sol)[1] / 2))

    # verbosity: if True, values of T_diff and C_diff and according edge will be printed
    verbose = kwargs.get('verbose', False)

    # iterate over each pairwise combi of variables, compute cross-rp, isrn and C,T measures
    # initialize arrays
    C_xys = np.zeros((n, n))
    C_yxs = np.zeros((n, n))
    T_xys = np.zeros((n, n))
    T_yxs = np.zeros((n, n))
    rrx = np.zeros(n)
    rry = np.zeros(n)
    rrxy = np.zeros((n,n))

    edges = []
    edges_ = []

    for i in range(n):
        for j in range(n):
            # choose time series
            if velocities_only:
                x = sol[:, i+n]
                y = sol[:, j+n]
            else:
                if sandwiched_couples:
                    x = sol[:, [2*i,2*i+1]]
                    y = sol[:, [2*j,2*j+1]]
                else:
                    x = sol[:, i]
                    y = sol[:, j]

            # compute the network
            net = InterSystemRecurrenceNetwork(x, y, threshold=th, metric=metric)

            rrxy[i,j] = net.cross_recurrence_rate()
            rrx[i], rry[i] = net.internal_recurrence_rates()

            # get the interesting metrics
            # - cross-clustering coefficient C_xy and C_yx
            # - cross-transitivity T_xy and T_yx
            C_xys[i, j] = net.cross_global_clustering_xy()
            C_yxs[i, j] = net.cross_global_clustering_yx()
            T_xys[i, j] = net.cross_transitivity_xy()
            T_yxs[i, j] = net.cross_transitivity_yx()

            edges.append([i, j])
            edges_.append([i, j])

    if savez:
        np.savez('network_arrays.npz', C_xys=C_xys, C_yxs=C_yxs, T_xys=T_xys, T_yxs=T_yxs, rrxy = rrxy, rrx=rrxy)

    # generate graph
    G = nx.DiGraph(edges)

    # generate graph
    G_ = nx.DiGraph(edges_)

    # generate graph from common edges, i.e. edges that are indicated by both T and C
    common_edges = common_elements(edges, edges_)
    common_G = nx.DiGraph(common_edges)

    return G, G_, common_G, C_xys, C_yxs, T_xys, T_yxs, rrx, rrxy

def compute_functional_network(sol, rr, **kwargs):
    """
    For a given time series sol, compute the inter-system-recurrence-network
    based on cross-clustering and cross-transitivity.
    - Iterative process: for each combi of two dofs:
        - generate isrn
        - compute cross-clustering coefficients, cross-transitivities and store in matrices C_xys, C_yxs and T_xys, T_yxs
    - generate two matrices C_diff = C_xys - C_yxs, T_diff = T_xys - T_yxs
    - create edge lists based on C_diff, T_diff and thresholds
    - generate common graph edges list by comparing C- and T- based graphs

    :param sol: time series data in [n_timesteps, 2*n], where n number of oscillators and sol has both displacements and velocities
    :param rr: recurrence rate to use for the computations. in rr = (rr_x, rr_y, rr_xy). suggested values: (0.3,0.3,0.2)
    :param kwargs: C_threshold and T_threshold. define a threshold below which entries in C_diff and T_diff will be considered as bi-directional edges
    :param kwargs: n: number of variables from the time series to use, if not the first half of variables in the
    time series
    :return: G based on C_diff, G_ based on T_diff and common_G which only contains common edges
    """
    #

    velocities_only = kwargs.get('velocities_only', False)
    sandwiched_couples = kwargs.get('sandwiched_couples', True)
    metric = kwargs.get('metric', 'euclidean')
    savez = kwargs.get('savez', True)

    # get number of variables from each time series to use
    # default = use first half, i.e. the positional ones
    n = kwargs.get('n', int(np.shape(sol)[1] / 2))

    # threshold for adding axes
    C_threshold = kwargs.get('C_threshold', 0)
    T_threshold = kwargs.get('T_threshold', 0)

    # verbosity: if True, values of T_diff and C_diff and according edge will be printed
    verbose = kwargs.get('verbose', False)

    # iterate over each pairwise combi of variables, compute cross-rp, isrn and C,T measures
    # initialize arrays
    C_xys = np.zeros((n, n))
    C_yxs = np.zeros((n, n))
    T_xys = np.zeros((n, n))
    T_yxs = np.zeros((n, n))
    epsilon = np.zeros((n,n))


    for i in range(n):
        for j in range(n):
            # choose time series
            if velocities_only:
                x = sol[:, i+n]
                y = sol[:, j+n]
            else:
                if sandwiched_couples:
                    x = sol[:, [2*i,2*i+1]]
                    y = sol[:, [2*j,2*j+1]]
                else:
                    x = sol[:, i]
                    y = sol[:, j]

            # compute the network
            net = InterSystemRecurrenceNetwork(x, y, recurrence_rate=rr, metric=metric)
            epsilon[i,j] = net.threshold

            # get the interesting metrics
            # - cross-clustering coefficient C_xy and C_yx
            # - cross-transitivity T_xy and T_yx
            C_xys[i, j] = net.cross_global_clustering_xy()
            C_yxs[i, j] = net.cross_global_clustering_yx()
            T_xys[i, j] = net.cross_transitivity_xy()
            T_yxs[i, j] = net.cross_transitivity_yx()

    # compute measure differences to create graphs
    C_diff = C_xys - C_yxs
    T_diff = T_xys - T_yxs

    if savez:
        np.savez('network_arrays.npz', C_xys=C_xys, C_yxs=C_yxs, T_xys=T_xys, T_yxs=T_yxs, C_diff=C_diff, T_diff=T_diff, epsilon=epsilon)

    # create an array of edges according to information in C_xy and C_yx
    edges = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            else:
                if C_diff[i, j] > C_threshold:
                    edges.append([j, i])
                elif C_diff[i, j] < -C_threshold:
                    edges.append([i, j])

                # optional: add bi-directional coupling when C_xy = C_yx
                else:
                    edges.append([i, j])
                    edges.append([j, i])

            # print for debugging
            if verbose:
                print(f'combi: {i} and {j}, C_diff = {C_diff[i, j]:.4f}, entry: {edges} ')

    # generate graph
    G = nx.DiGraph(edges)

    # create an array of edges according to information in T_xy and T_yx
    edges_ = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            else:
                if T_diff[i, j] > T_threshold:
                    edges_.append([j, i])
                elif T_diff[i, j] < -T_threshold:
                    edges_.append([i, j])

                # # optional: add bi-directional coupling when C_xy = C_yx
                # else:
                #     edges_.append([i, j])
                #     edges_.append([j, i])

            # print for debugging
            if verbose:
                print(f'combi: {i} and {j}, Tdiff = {T_diff[i, j]:.4f}, entry: {edges_} ')

    # generate graph
    G_ = nx.DiGraph(edges_)

    # generate graph from common edges, i.e. edges that are indicated by both T and C
    common_edges = common_elements(edges, edges_)
    common_G = nx.DiGraph(common_edges)

    return G, G_, common_G, T_diff, C_diff, C_xys, C_yxs, T_xys, T_yxs



def compute_functional_network_by_matrix(sol, rr, **kwargs):
    """
    Compute C_xys,T_xys for the data array sol.
    Use matrix methods.
    """

    metric = kwargs.get('metric','euclidean')

    n_time,n_nodes = sol.shape

    C_xys = np.zeros((n_nodes,n_nodes))
    T_xys = np.zeros((n_nodes,n_nodes))

    for i in range(n_nodes):
        series_x = sol[:,i]
        for j in range(i+1,n_nodes):
            if j != i:
                series_y = sol[:,j]
                net = InterSystemRecurrenceNetwork(series_x, series_y, recurrence_rate=rr, metric=metric)
                AX = net.rp_x - np.eye(net.rp_x.shape[0])
                AY = net.rp_y - np.eye(net.rp_y.shape[0])
                CRXY = net.crp_xy
                CRYX = CRXY.T
                kXY = np.sum(CRXY[i])
                C_xys[i,j] = (CRYX @ AY @ CRXY)[i,i] / ( kXY*(kXY-1) )
                T_xys[i,j] = np.trace(CRYX @ AY @ CRXY) / ( np.sum(CRXY@CRYX) - np.sum(CRXY@CRYX) )
    for i in range(n_nodes):
        for j in range(i+1):
            C_xys[j,i] = C_xys[i,j]
            T_xys[j,i] = T_xys[i,j]

    return C_xys, T_xys


def network_computation(data_directory_main, result_directory_main, **kwargs):
    """
    Compute hybrid isrn network for all parameter variations and time series within a given data_directory_main.
    Data structure: in data_directory_main/<parameter variations>/<time series in modal and cart. coordinates>
    Result structure will be the same: result_directory_main/<parameter variations>/<isrn networks>
    :param data_directory_main:
    :param result_directory_main:
    :kwargs: time_series_length in s, will be /by 0.05 to get number of samples
    :return:

    """

    time_series_length = kwargs.get('time_series_length', -1)
    noise_level = kwargs.get('noise_level', 0)
    n = kwargs.get('n', 10)

    # convert time series length to number of samples. dt = 0.05
    if time_series_length > -1:
        time_series_length = int(time_series_length/0.05)

    # define desired recurrence rate, as recommended in Feldhoff.2012
    rr = (0.03, 0.03, 0.02)

    # define thresholds
    C_threshold = 0.02
    T_threshold = C_threshold

    # loop over all m-variation folders in data directory
    # this list contains t, and the cartesian and modal data for each dynamical state
    for m_variation_directory in os.listdir(data_directory_main):
        full_path = os.path.join(data_directory_main, m_variation_directory)
        if not os.path.isdir(full_path):
            continue

        # skip the time and mvars arrays
        if m_variation_directory.endswith('.npy'):
            continue
        else:
            # obtain data storage location
            data_directory = os.path.join(data_directory_main, m_variation_directory)

            # generate storing location
            result_directory_m = os.path.join(result_directory_main, m_variation_directory)
            os.mkdir(result_directory_m)

            for dataset in os.listdir(data_directory):

                # skip jpegs such that every data set is only loaded once
                if dataset.endswith('.jpg'):
                    continue

                # generate hybrid network for every set of data
                else:
                    data_name = dataset[:-6]
                    print(f'- starting network computation for {data_name}.')

                    # create a sub-directory for networks from each data set
                    result_directory = os.path.join(result_directory_m, data_name)
                    os.mkdir(result_directory)

                    sol_c = np.load(os.path.join(data_directory, f'{data_name}_c.npy'))

                    # specify time series settings
                    # - use only section of time series
                    sol_c_ = sol_c[:time_series_length, :n]

                    # - if defined, add noise
                    if noise_level == 1:
                        from utils.add_noise_to_timeseries import add_noise_to_timeseries
                        sol_c_ = add_noise_to_timeseries(sol_c_, mean=0, std_mode='from_ts')

                    _, _, G = compute_functional_network(sol_c_, rr,
                                                         n=n,
                                                         C_threshold=C_threshold,
                                                         T_threshold=T_threshold)

                    nx.write_edgelist(G, os.path.join(result_directory, 'G_cartesian'))


def align_phase(ts1, ts2):
    """
    Adjusts ts1 and ts2 such that (ts1 - ts2) is between -pi and pi.
    This is done by adding/subtracting multiples of 2*pi to the difference.
    """
    # Calculate the raw difference
    diff = ts1 - ts2
    
    # Map the difference to the range [-pi, pi]
    # The formula: ((diff + pi) % (2*pi)) - pi
    adjusted_diff = ((diff + np.pi) % (2 * np.pi)) - np.pi
    
    # To 'align' the series, we can adjust ts1 to match ts2's reference frame
    ts1_aligned = ts2 + adjusted_diff
    
    return ts1_aligned, ts2

def align_multiple_series(data, ref_idx=0):
    """
    Aligns multiple time series to a reference series.
    
    Parameters:
    data: np.ndarray of shape (time_steps, num_series)
    ref_idx: index of the series to use as the reference
    """
    # Extract reference column and reshape for broadcasting
    ref_series = data[:, [ref_idx]] 
    
    # Calculate difference between all series and the reference
    diff = data - ref_series
    
    # Apply the wrapping logic: ((diff + pi) % 2pi) - pi
    # This finds the shortest angular distance for every point
    wrapped_diff = ((diff + np.pi) % (2 * np.pi)) - np.pi
    
    # Reconstruct the aligned data
    aligned_data = ref_series + wrapped_diff
    
    return aligned_data

# if __name__ == '__main__':

#     # data directory
#     data_directory_main = 'data/all_time_series'

#     # define storage directory for the results
#     result_directory_main = 'results/all_time_series'
#     os.mkdir(result_directory_main)

#     network_computation(data_directory_main,
#                         result_directory_main,
#                         time_series_length=10,
#                         noise_level=0)
    

# for dataset in os.listdir(data_directory_main):
#         dataset_path = os.path.join(data_directory_main, dataset)
#         if not os.path.isdir(dataset_path):
#             continue  # skip files like .DS_Store
#         # process the dataset folder


