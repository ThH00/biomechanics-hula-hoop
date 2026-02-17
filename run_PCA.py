import json
from pathlib import Path
import pprint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utilities_hula_hoop import data_to_array, plot_PCA_modes_by_segment

if __name__ == "__main__":

    VERBOSE = 1 # 0: no printing or showing plots
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
    X_PCA_hftc = X_pca

    fig = plot_PCA_modes_by_segment(eigenvectors,quantities)
    plt.savefig(OUT_DIR/'PCA.pdf', dpi=400)

    if VERBOSE:
        print(f"\nPCA Plot saved as {str(OUT_DIR/'PCA.pdf')}")
        print("\nExplained Variance Ratios:", explained_variance_ratio)
        plt.show()

