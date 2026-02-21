""" 
Created by Theresa E. Honein and Chrystal Chern as part of the
accompanying code and data for the paper, submitted in 2026, to the
Proceedings of the Royal Society A, "The Biomechanics of Hula Hooping"
by C. Chern, T. E. Honein, and O. M. O'Reilly.

Licensed under the GPLv3. See LICENSE in the project root for license information.

February 20, 2026

"""

import json
from pathlib import Path
import pprint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utilities import (data_to_array,
                        plot_PCA_modes_by_segment,
                        plot_PCA_variance_ratios,
                        plot_PCA_phase_portait,
                        plot_PCA_FFT
                        )

if __name__ == "__main__":

    VERBOSE = 1 # 0: no printing or plot showing
                # 1: basic overview and plots
                # 2: overview, detailed descriptions, and plots

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

    # Step 1: Gather data array with desired quantities
    quantities={
        'femur':['wx','wy','wz'],
        'tibia':['wx','wy','wz'],
        'metatarsal':['wx','wy','wz'],
        'hoop':['wxy','psidot'],
    }
    X = data_to_array(data_dict=data_dict, quantities=quantities)
    if VERBOSE:
        print("\nPerforming PCA on data array constructed "
                "from measured time series:")
        for sensor,qset in quantities.items():
            print(f"{sensor:} {qset}")
    if VERBOSE >= 2:
        print("Data array shape:", np.shape(X))

    # Step 2: Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: Apply PCA
    pca = PCA(n_components=np.shape(X)[1])
    X_pca = pca.fit_transform(X_scaled)
    # Get the eigenvectors (principal directions)
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    explained_variance_ratio = pca.explained_variance_ratio_
    fig_PCA = plot_PCA_modes_by_segment(eigenvectors,quantities)
    fig_PCA.savefig(OUT_DIR/'PCA.pdf')

    fig_var = plot_PCA_variance_ratios(explained_variance_ratio)
    fig_var.savefig(OUT_DIR/'PCA_variance_ratios.pdf')

    fig_phase = plot_PCA_phase_portait(X_pca)
    fig_phase.savefig(OUT_DIR/'PCA_phase_portrait.pdf')

    time = data_dict['hoop']['time']
    fig_FFT = plot_PCA_FFT(X_pca,dt=time[1]-time[0])
    fig_FFT.savefig(OUT_DIR/'PCA_FFT.pdf')

    if VERBOSE:
        print("\nExplained Variance Ratios:", explained_variance_ratio)
        print(f"\nPCA Plot saved as {str(OUT_DIR/'PCA.pdf')}")
        print(f"Variance Ratio Plot saved as {str(OUT_DIR/'PCA_variance_ratios.pdf')}")
        print(f"Phase Portrait Plot saved as {str(OUT_DIR/'PCA_phase_portrait.pdf')}")
        print(f"PCA FFT saved as {str(OUT_DIR/'PCA_FFT.pdf')}")
        plt.show()

