import json
import pprint
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from utilities import run_network, plot_network, SYMDICT

if __name__ == "__main__":

    COMPUTE_NEW = True # Compute new networks

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
            target_nodes = [f"{SYMDICT['wxy']},hoop"],
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
            target_nodes = [f"{SYMDICT['psidot']},hoop"],
            width_scale=50.0,
            rr=(0.03,0.03,0.02), # recurrence rates
            out_dir=OUT_DIR,
            verbose=VERBOSE,
            savez=False,
            save_arrays=True,
            # ntime=400, # to do a quicker analysis, uncomment to analyze a portion
        ),
    }

    # Compute Networks
    networks = {}
    mappings = {}
    target_node_lists = {}

    if COMPUTE_NEW:
        if (OUT_DIR/'network_dictionaries.npz').exists() or False:
            if VERBOSE:
                print("Existing data found. Appending and Amending.")
            network_dictionaries = np.load(OUT_DIR/'network_dictionaries.npz',allow_pickle=True)
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

        np.savez(OUT_DIR/'network_dictionaries.npz', networks=networks, mappings=mappings, target_node_lists=target_node_lists)
    
    else:
        if (OUT_DIR/'network_dictionaries.npz').exists():
            if VERBOSE:
                print("Existing data found. Uploading.")
            network_dictionaries = np.load(OUT_DIR/'network_dictionaries.npz',allow_pickle=True)
            networks = network_dictionaries['networks'].item()
            mappings = network_dictionaries['mappings'].item()
            target_node_lists = network_dictionaries['target_node_lists'].item()
        else:
            if VERBOSE:
                print("No existing data found.")

    
    # Plot Networks
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


