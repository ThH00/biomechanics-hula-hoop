# packages
import numpy as np
from network_computation import compute_functional_network
import pickle

# loading data
data = np.load("/Users/theresahonein/Desktop/terryhonein/Research-HulaHoop/experiments/data_experiment_5.npy")
data = data.T

# computing network
G, G_, common_G = compute_functional_network(data, (0.02,0.02,0.01), C_threshold=0.8, T_threshold=0.8, n=np.shape(data)[1] )

import pickle

# Save
with open("networks4.pkl", "wb") as f:
    pickle.dump((G, G_, common_G), f)

# Load later
with open("networks.pkl", "rb") as f:
    G, G_, common_G = pickle.load(f)


print('done')