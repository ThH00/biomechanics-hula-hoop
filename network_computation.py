# -*- coding: utf-8 -*-
""" 
Modified version of original work by Charlotte Geier,
Hamburg University of Technology
charlotte.geier@tuhh.de

File adapted by Theresa E. Honein and Chrystal Chern as part of the
accompanying code and data for the paper, submitted in 2026, to the
Proceedings of the Royal Society A, "The Biomechanics of Hula Hooping"
by C. Chern, T. E. Honein, and O. M. O'Reilly.

The functions contained in this file are adapted from the accompanying
code for the paper "Exploring localization in nonlinear oscillator systems
through network-based predictions" by C. Geier and N. Hoffmann published
in Chaos 35 (5) 2025 doi: 10.1063/5.0265366 .
Available at https://arxiv.org/abs/2407.05497

Licensed under the GPLv3. See LICENSE in the project root for license information.

20.02.2026


-------- Original Comments by C. Geier and N. Hoffmann -------

main file for functional network computation

Part of the accompanying code for the paper "Exploring localization in nonlinear
oscillator systems through network-based predictions" by C. Geier and N. Hoffmann
published in Chaos 35 (5) 2025 doi: 10.1063/5.0265366 .
Available at https://arxiv.org/abs/2407.05497

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
import numpy as np
import tqdm


def common_elements(list1, list2):
    """
    Find the common elements of two lists. Especially also helpful when looking at lists of lists,
    where intersection() fails.
    :param list1: a list
    :param list2: a second list
    :return: common elements of the two lists
    """
    result = []
    for element in list1:
        if element in list2:
            result.append(element)
    return result


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

    :param sol: time series data in [n_timesteps, 2*n] (sandwiched_couples=True)
                or [n_timesteps, n, 2] (sandwiched_couples=False), where n number of nodes
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

    # verbosity: if 1 or True, progress bar will be shown
    #            if >=2, recurrence network calculations will be printed 
    #            if >=2, values of T_diff and C_diff and according edge will be printed
    verbose = kwargs.get('verbose', False)

    # iterate over each pairwise combi of variables, compute cross-rp, isrn and C,T measures
    # initialize arrays
    C_xys = np.zeros((n, n))
    C_yxs = np.zeros((n, n))
    T_xys = np.zeros((n, n))
    T_yxs = np.zeros((n, n))
    epsilon = np.zeros((n,n))

    if verbose:
        iterator = tqdm.tqdm(range(n), desc=f"Progress over {n} nodes")
    else:
        iterator = range(n)

    for i in iterator:
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
            net = InterSystemRecurrenceNetwork(x, y, recurrence_rate=rr, metric=metric, silence_level=3-verbose)
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
            if verbose>=2:
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
            if verbose>=2:
                print(f'combi: {i} and {j}, Tdiff = {T_diff[i, j]:.4f}, entry: {edges_} ')

    # generate graph
    G_ = nx.DiGraph(edges_)

    # generate graph from common edges, i.e. edges that are indicated by both T and C
    common_edges = common_elements(edges, edges_)
    common_G = nx.DiGraph(common_edges)

    return G, G_, common_G, T_diff, C_diff, C_xys, C_yxs, T_xys, T_yxs




