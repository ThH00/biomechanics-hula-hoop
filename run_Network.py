""" 
Created by Theresa E. Honein and Chrystal Chern as part of the
accompanying code and data for the paper, submitted in 2026, to the
Proceedings of the Royal Society A, "The Biomechanics of Hula Hooping"
by C. Chern, T. E. Honein, and O. M. O'Reilly.

Licensed under the GPLv3. See LICENSE in the project root for license information.

To compute new networks of different body and hoop motion timeseries, or with
different parameters such as recurrence rates and window sizes, modify the
network_parameters dictionary.

Be sure that COMPUTE_NEW_NETWORKS and COMPUTE_NEW_WINDOWS are set to True
when computing new networks.

February 20, 2026

"""

import json
import pprint
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from utilities import (run_network,
                       plot_network,
                       SYMDICT,
                       data_to_array_by_quantity,
                       get_time_networks,
                       plot_heatmaps
                      )

if __name__ == "__main__":

    COMPUTE_NEW_NETWORKS = False # Compute new networks (full)
    COMPUTE_NEW_WINDOWS = False # Compute new networks (windowed)

    VERBOSE = 1 # 0: no printing or showing plots
                # 1: basic overview and plots
                # 2: overview, detailed descriptions, debugging, plots

    # Load data and define output directory
    with open("data/hula_hooping_records.json", 'r') as f:
        data_dict = json.load(f)
    OUT_DIR = Path("plots")
    OUT_DIR.mkdir(exist_ok=True)

    if VERBOSE >=2:
        print("Data Loaded.")
        print("\nSensors Included:", list(data_dict.keys()))
        print(f"{len(list(data_dict['hoop'].keys()))} Quantities Included:")
        print(list(data_dict['hoop'].keys()))
        print("\nData Dictionary Structure:")
        print(f"{pprint.pformat(data_dict, depth=2)[:210]}\n...")

    # Network Analysis

    # List Parameters for Networks
    network_parameters = {
        'w_xy': dict(
            quantities={
                'hoop':[['time','wxy']],
                'femur':[['time','wx'], ['time','wy'], ['time','wz']],
                'tibia':[['time','wx'], ['time','wy'], ['time','wz']],
                'metatarsal':[['time','wx'], ['time','wy'], ['time','wz']],
            },
            target_node = f"{SYMDICT['wxy']},hoop",
            width_scale=50.0,
            rr=(0.03,0.03,0.02), # recurrence rates
            out_dir=OUT_DIR,
            verbose=VERBOSE,
            savez=False,
            save_arrays=True,
            # ntime=400, # to do a quicker analysis, uncomment to analyze a portion
        ),
        'psidot': dict(
            quantities={
                'hoop':[['time','psidot']],
                'femur':[['time','wx'], ['time','wy'], ['time','wz']],
                'tibia':[['time','wx'], ['time','wy'], ['time','wz']],
                'metatarsal':[['time','wx'], ['time','wy'], ['time','wz']],
            },
            target_node = f"{SYMDICT['psidot']},hoop",
            width_scale=50.0,
            rr=(0.03,0.03,0.02), # recurrence rates
            out_dir=OUT_DIR,
            verbose=VERBOSE,
            savez=False,
            save_arrays=True,
            # ntime=400, # to do a quicker analysis, uncomment to analyze a portion
        ),
    }


    # Compute Overall Networks
    networks = {}
    mappings = {}
    target_node_lists = {}

    if COMPUTE_NEW_NETWORKS:
        if Path("data/network_dictionaries.npz").exists() or False:
            if VERBOSE:
                print("Existing data found. Appending and Amending.")
            network_dictionaries = np.load("data/network_dictionaries.npz",allow_pickle=True)
            networks = network_dictionaries['networks'].item()
            mappings = network_dictionaries['mappings'].item()
            target_node_lists = network_dictionaries['target_node_lists'].item()
            if VERBOSE:
                print(f"Existing configurations: {[title for title in networks.keys()]}")
            
        network_quantities = ['G', 'G_', 'common_G', 'T_diff', 'C_diff', 'C_xys', 'C_yxs', 'T_xys', 'T_yxs']

        for title,parameters in network_parameters.items():
            if VERBOSE:
                print(f"\nNow appending/amending configuration: {title}")
            network,mapping,target_nodes = run_network(data_dict,title,**parameters)

            networks[title] = {q: network[i] for i,q in enumerate(network_quantities)}
            mappings[title] = mapping
            target_node_lists[title] = target_nodes

        np.savez("data/network_dictionaries.npz", networks=networks, mappings=mappings, target_node_lists=target_node_lists)
    
    else:
        if Path("data/network_dictionaries.npz").exists():
            if VERBOSE:
                print("Existing data found. Uploading.")
            network_dictionaries = np.load("data/network_dictionaries.npz",allow_pickle=True)
            networks = network_dictionaries['networks'].item()
            mappings = network_dictionaries['mappings'].item()
            target_node_lists = network_dictionaries['target_node_lists'].item()
        else:
            if VERBOSE:
                print("No existing data found.")

    
    # Plot Overall Networks
    for title,parameters in network_parameters.items():

        for coeff_xy in ['C_xys','T_xys']:
            c_xy = networks[title][coeff_xy]
            diff_xy = c_xy-c_xy.T
            plot_title = f"{title}_{coeff_xy.replace('xys','diff')}"
            if VERBOSE:
                print(f"\nNow plotting {plot_title}")

            if VERBOSE>=2:
                print(f"\n{title} node mappings and target node:")
                print(f"Mappings:  {mappings[title]}")
                print(f"Target Node: {target_node_lists[title]}")

            fig = plot_network(diff_xy,
                        mappings[title],
                        target_node_lists[title],
                        width_scale=parameters.get('width_scale',5.0),
                        diffs=parameters.get('diffs',True),
                        draw_from_target_edges=parameters.get('draw_from_target_edges',False),
                        draw_to_target_edges=parameters.get('draw_to_target_edges',True),
                        draw_no_target_edges=parameters.get('draw_no_target_edges',False),
                        verbose=parameters.get('verbose',False),
                        title = plot_title)
            fig.savefig(OUT_DIR/f"{plot_title}.png", transparent=True)
            if VERBOSE:
                plt.show()
                print(f"Saved network plot: {OUT_DIR}/{plot_title}.png")


    # Compute Windowed Networks
    WINDOW_SIZE = 190
    STEP_SIZE = 10
    if COMPUTE_NEW_WINDOWS:
        for title,parameters in network_parameters.items():
            if VERBOSE:
                print(f"\nComputing windowed networks for {title}, window={WINDOW_SIZE}, step={STEP_SIZE}")
            quantities = parameters['quantities']

            data = data_to_array_by_quantity(data_dict,
                                    quantities=quantities)
            
            network_windows = get_time_networks(data=data,
                                                rr=parameters['rr'],
                                                window_size=WINDOW_SIZE,
                                                step_size=STEP_SIZE)
            np.save(f"data/{title}_windows.npy", network_windows)
            if VERBOSE:
                print(f"Saved windowed network data: {OUT_DIR}/{title}_windows.npy")


    # Plot Windowed Network Heatmaps
    time=data_dict['hoop']['time']

    for title,parameters in network_parameters.items():
        if VERBOSE:
            print(f"\nPlotting heatmaps for {title}, window={WINDOW_SIZE}, step={STEP_SIZE}")
        quantities = parameters['quantities']
        target_node = parameters['target_node']
        node_labels = [f"{SYMDICT[q[1]]},{s}" for s,qsets in quantities.items() for q in qsets]

        network_windows = np.load(f"data/{title}_windows.npy")
        figs = plot_heatmaps(network_windows,
                node_labels,
                target_node,
                time=time,
                heatmap_max=parameters.get('heatmap_max',0.5),
                window_size=WINDOW_SIZE,
                step_size=STEP_SIZE,
                plot_filename_prefix=OUT_DIR/title,
                return_figs=True)
        if VERBOSE:
            plt.show()
            print(f"Saved heatmap: {OUT_DIR}/{title}_..._{WINDOW_SIZE}.pdf")