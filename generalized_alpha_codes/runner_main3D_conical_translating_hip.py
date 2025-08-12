import os
import argparse
from datetime import datetime
from main3D_conical_translating_hip import Simulation 
import sys
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ARGS_NAMES = ['ntime', 'mus', 'muk', 'eN', 'eF', 'maxleaves', 'z0', 'dphi0']

# Generate timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
outputs_dir = f"outputs/experiment{timestamp}"
input_file = "inputs_conical_translating_hip"

# Check if the input file exists
if not os.path.isfile(input_file):
    print(f"Input file '{input_file}' not found.")
    exit(1)


def do_run(args, args_list):
    print(f"Running with args: {[(f'{ARGS_NAMES[i]}: {arg}') for i, arg in enumerate(args_list)]}")
    
    # Create output directory for this run
    run_output_dir = f"{outputs_dir}/{'_'.join([(f'{ARGS_NAMES[i]}{arg}') for i, arg in enumerate(args_list)])}"
    os.makedirs(run_output_dir, exist_ok=True)

    # Convert args_list to individual arguments
    ntime, mu_s, mu_k, eN, eF, max_leaves, z0, dphi0 = args_list

    # Run the simulation
    sim = Simulation(
        ntime=int(ntime),
        mu_s=mu_s,
        mu_k=mu_k,
        eN=eN,
        eF=eF,
        max_leaves=int(max_leaves),
        z0=z0,
        dphi0=dphi0,
        output_path=run_output_dir
    )

    # Redirect output (stdout) to file
    stdout_path = os.path.join(run_output_dir, "stdout.out")
    with open(stdout_path, 'w') as f:
        old_stdout = sys.stdout
        sys.stdout = f
        try:
            sim.solve_A()
        finally:
            sys.stdout = old_stdout

    print(f"Completed run. Output in: {run_output_dir}")


def read_input_file(filepath):
    """Read all datasets from the input file."""
    datasets = []
    with open(filepath, 'r') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # skip empty lines
            values = line.split()
            if len(values) != 8:
                raise ValueError(f"Line {line_number}: must contain exactly 8 space-separated values.")
            datasets.append([
                int(values[0]),         # ntime
                float(values[1]),       # mu_s
                float(values[2]),       # mu_k
                float(values[3]),       # eN
                float(values[4]),       # eF
                int(values[5]),         # max_leaves
                float(values[6]),       # z0
                float(values[7])        # dphi0
            ])
    return datasets


if __name__ == "__main__":
    all_datasets = read_input_file(input_file)
    for dataset in all_datasets:
        do_run(args=None, args_list=dataset)