
import os
import argparse
from datetime import datetime
from main3D_conical_translating_hip import Simulation 
import sys
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

test = os.getcwd()


# def parse_args():
#     parser = argparse.ArgumentParser(description="Run hoop simulation with parameters.")
#     parser.add_argument('--mu_s', type=float, default=1.0, help='Static friction coefficient')
#     parser.add_argument('--mu_k', type=float, default=0.8, help='Kinetic friction coefficient')
#     parser.add_argument('--eN', type=float, default=0.0, help='Normal restitution')
#     parser.add_argument('--eF', type=float, default=0.0, help='Friction restitution')
#     parser.add_argument('--ntime', type=int, default=100, help='Number of time steps')
#     parser.add_argument('--max_leaves', type=int, default=5, help='Max number of leaves in solver')
#     parser.add_argument('--z0', type=float, default=-10, help='Initial hoop height')
#     parser.add_argument('--dphi0', type=float, default=2, help='Initial phidot of hoop')
#     parser.add_argument('--output_path', '-o', type=str, help='Output path (optional)')
    
#     args = parser.parse_args()

#     return args.ntime, args.mu_s, args.mu_k, args.eN, args.eF, args.max_leaves, args.z0, args.dphi0, args.output_path


ARGS_NAMES = ['ntime', 'mus', 'muk', 'eN','eF', 'maxleaves', 'z0', 'dphi0']

# Generate timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
outputs_dir = f"outputs/experiment{timestamp}"
source_file = "main3D_conical_translating_hip"
input_file = "inputs_conical_translating_hip"

test = os.getcwd()

# Check if the input file exists
if not os.path.isfile(input_file):
    print(f"Input file '{input_file}' not found.")
    exit(1)


def do_run(args, args_list):
    print(f"Running with args: {[(f'{ARGS_NAMES[i]}: {arg}') for i, arg in enumerate(args_list)]}")
    
    # Create output directory
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
        dphi0 =dphi0,
        output_path=run_output_dir
    )

    # Redirect output (stdout) to file
    stdout_path = os.path.join(run_output_dir, "stdout.out")
    with open(stdout_path, 'w') as f:
        # Optional: redirect print to file
        import sys
        old_stdout = sys.stdout
        sys.stdout = f
        try:
            sim.solve_A()
        finally:
            sys.stdout = old_stdout

    print(f"Completed run. Output in: {run_output_dir}")

def read_input_file(filepath):
    with open(filepath, 'r') as f:
        line = f.readline().strip()
        values = line.split()
        if len(values) != 8:
            raise ValueError("Input file must contain exactly 8 space-separated values.")
        # Convert values to the proper types
        return [
            int(values[0]),         # ntime
            float(values[1]),       # mu_s
            float(values[2]),       # mu_k
            float(values[3]),       # eN
            float(values[4]),       # eF
            int(values[5]),         # max_leaves
            float(values[6]),       # z0
            float(values[7])        # dphi0
        ]



if __name__ == "__main__":
    args_list = read_input_file(input_file)
    do_run(args=None, args_list=args_list)