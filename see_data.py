""" 
Created by Theresa E. Honein and Chrystal Chern as part of the
accompanying code and data for the paper, submitted in 2026, to the
Proceedings of the Royal Society A, "The Biomechanics of Hula Hooping"
by C. Chern, T. E. Honein, and O. M. O'Reilly.

Licensed under the GPLv3. See LICENSE in the project root for license information.

20.02.2026

"""

import json
from pathlib import Path
import pprint
from utilities import plot_time_histories

if __name__ == "__main__":

    VERBOSE = 1 # 0: no printing
                # 1: basic overview
                # 2: overview, detailed descriptions

    WRITE_PLOTS = False # To save plots

    # Load data and define output directory
    with open("data/hula_hooping_records.json", 'r') as f:
        data_dict = json.load(f)
    if WRITE_PLOTS:
        OUT_DIR = Path("plots")
        OUT_DIR.mkdir(exist_ok=True)

    if VERBOSE >=2:
        print("Data Loaded.")
        print("\nSensors Included:", list(data_dict.keys()))
        print(f"{len(list(data_dict['hoop'].keys()))} Quantities Included:")
        print(list(data_dict['hoop'].keys()))
        print("\nData Dictionary Structure:")
        print(f"{pprint.pformat(data_dict, depth=2)[:210]}\n...")

    # Plot some time series
    sensor_labels = {
    'hoop': 'Hoop', 
    'femur': 'Femur',
    'tibia': 'Tibia',
    'metatarsal': 'Metatarsal'
    }
    
    # Angular velocities
    if VERBOSE:
        print(f"\nPlotting angular velocities.")
    quantities = ['wx','wy','wz']
    time = data_dict['hoop']['time']
    data_dict_wxyz = {sensor: {q:v for q,v in serieses.items() if q in quantities}
                      for sensor,serieses in data_dict.items()}
    fig = plot_time_histories(sensor_labels, data_dict_wxyz, time, title="", one_per=True)
    fig.show()
    if WRITE_PLOTS:
        fig.write_image(OUT_DIR/"angular_velocities.pdf")
        if VERBOSE:
            print(f"Angular velocities plot saved as {str(OUT_DIR/"angular_velocities.pdf")}")
    
    # Euler derivatives
    if VERBOSE:
        print(f"\nPlotting euler derivatives.")
    quantities = ['phidot','thetadot','psidot']
    data_dict_ptpdot = {sensor: {q:v for q,v in serieses.items() if q in quantities}
                      for sensor,serieses in data_dict.items()}
    fig = plot_time_histories(sensor_labels, data_dict_ptpdot, time, title="", one_per=True)
    fig.show()
    if WRITE_PLOTS:
        fig.write_image(OUT_DIR/"euler_derivatives.pdf")
        if VERBOSE:
            print(f"Euler derivatives plot saved as {str(OUT_DIR/"euler_derivatives.pdf")}")