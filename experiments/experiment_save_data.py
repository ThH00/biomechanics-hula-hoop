"""
Usage:
    python experiment_save_data.py
    python experiment_save_data.py <input directory> <sensor file number>

Example: use the default Experiment 5 folder and the sensor files XX_20250903_203926.csv
    python experiment_save_data.py
Equivalent: explicity specify it
    python experiment_save_data.py 'Users/theresahonein/Desktop/terryhonein/Research-HulaHoop/experiments/Hula Hoop/2025-09-03 Experiment 5/2025-09-03 Euler Angles/' '20250903_203926'
"""

import sys
import numpy as np
from pathlib import Path
from process_movella import load_movella, get_position
from utilities_hula_hoop import get_steady_hooping_interval, get_fixed_frame_acceleration
import json


if __name__ == "__main__":
    print("Python executable:", sys.executable)

    # Directories and file numbers
    if len(sys.argv) > 1:
        if len(sys.argv) != 3:
            raise SyntaxError("Exactly two arguments must be given: input folder and sensor file number")
        IN_DIR = Path(sys.argv[1])
        sensor_file_number = sys.argv[2]
    else:
        IN_DIR = Path("/Users/theresahonein/Desktop/terryhonein/Research-HulaHoop/experiments/Hula Hoop/2025-09-03 Experiment 5/2025-09-03 Euler Angles/")
        sensor_file_number = '20250903_203926'

    OUT_DIR = Path("out")
    if not OUT_DIR.exists():
        OUT_DIR.mkdir()

    # Populate data dictionary with raw data
    sensor_ids = ['OR','OL','IT','IL','IB']
    raw_quantities = ['time','ax','ay','az','phi','theta','psi','wx','wy','wz']
    data_dict = {s:{} for s in sensor_ids}
    lead_time_val = 0
    for s in sensor_ids:
        file = IN_DIR / f"{s}_{sensor_file_number}.csv"  # e.g. IN_DIR / "OR_20250903_203926.csv"
        sensor_data = load_movella(file, lead_time=lead_time_val)
        for i,q in enumerate(raw_quantities):
            data_dict[s][q] = sensor_data.T[i]

    # time step
    dt = data_dict['OR']['time'][1]-data_dict['OR']['time'][0]

    # Get steady hooping interval
    OR_groups, OR_averages = get_steady_hooping_interval(data_dict['OR']['psi'], dt=dt, threshold=0.55)
    slice_start = input("Enter START of steady interval or press enter (default: 1500):  ")
    slice_end = input("Enter END of steady interval or press enter (default: 2000):  ")
    if slice_start == "":
        slice_start = 1500
    if slice_end == "":
        slice_end = 2000
    active_slice = np.arange(int(slice_start), int(slice_end))

    # Trim to active slice
    for s in sensor_ids:
        for q in raw_quantities:
            data_dict[s][q] = data_dict[s][q][active_slice]


    # Get displacements and velocities
    quantities_local = ['ax','ay','az','psi','theta','phi']
    quantities_fixed = ['Ax','Ay','Az']
    quantities_pos = ['dx','dy','dz','vx','vy','vz']
    for s in sensor_ids:
        fixed_frame_data = get_fixed_frame_acceleration(*(data_dict[s][q] for q in quantities_local))
        for i,q in enumerate(quantities_fixed):
            data_dict[s][q] = fixed_frame_data[i]
        position_data = get_position(data_dict[s]['time'],data_dict[s]['Ax'],data_dict[s]['Ay'],data_dict[s]['Az'],degree=5,initial=0)
        for i,q in enumerate(quantities_pos):
            data_dict[s][q] = position_data[i]

    # Make all arrays into lists, to fix numpy compatibility between environments
    for s,q in data_dict.items():
        for k,v in q.items():
            data_dict[s][k] = v.tolist()

    # Save as JSON
    with open(f"data_{IN_DIR.name}_{sensor_file_number}.json", "w") as f:
        json.dump(data_dict, f)

