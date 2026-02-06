import os
import numpy as np

folder_cc = "/Users/theresahonein/Desktop/terryhonein/Research-HulaHoop/network-methods-code/2024-code-ExploringLocalizationInNonlinearOscillators"
# folder_cc = "/Users/theresahonein/Desktop/terryhonein/Research-HulaHoop/network-methods-code/2024-code-ExploringLocalizationInNonlinearOscillators/chrystal_rotation_results"
folder_th = "/Users/theresahonein/Desktop/terryhonein/Research-HulaHoop/network-methods-code/2024-code-ExploringLocalizationInNonlinearOscillators/theresa_rotation_results"

################################
## Comparing data arrays #######
################################


# data_cc = np.load(os.path.join(folder_cc, "psidot.npz"))
# data_cc = np.load(os.path.join(folder_cc, "data_cc_w.npy"))
data_cc = np.load("/Users/theresahonein/Desktop/terryhonein/Research-HulaHoop/network-methods-code/2024-code-ExploringLocalizationInNonlinearOscillators/data_test.npy")
data_th = np.load(os.path.join(folder_th, "data.npy"))

print("Shapes of data arrays:")
print("data_cc shape:", data_cc.shape)
print("data_th shape:", data_th.shape)

# comparing hoop data
print("Max absolute difference for hoop data:")
print(np.max(np.abs(data_cc[:,0,0]-data_th[:,0])))
print(np.max(np.abs(data_cc[:,0,1]-data_th[:,1])))

# comparing femur data
print("Max absolute difference for femur data:")
print(np.max(np.abs(data_cc[:,1,0]-data_th[:,2])))
print(np.max(np.abs(data_cc[:,1,1]-data_th[:,3])))

print(np.max(np.abs(data_cc[:,2,0]-data_th[:,4])))
print(np.max(np.abs(data_cc[:,2,1]-data_th[:,5])))

print(np.max(np.abs(data_cc[:,3,0]-data_th[:,6])))
print(np.max(np.abs(data_cc[:,3,1]-data_th[:,7])))

# comparing tibia data
print("Max absolute difference for tibia data:")
print(np.max(np.abs(data_cc[:,4,0]-data_th[:,8])))
print(np.max(np.abs(data_cc[:,4,1]-data_th[:,9])))

print(np.max(np.abs(data_cc[:,5,0]-data_th[:,10])))
print(np.max(np.abs(data_cc[:,5,1]-data_th[:,11])))

print(np.max(np.abs(data_cc[:,6,0]-data_th[:,12])))
print(np.max(np.abs(data_cc[:,6,1]-data_th[:,13])))

# comparing cuneiform data
print("Max absolute difference for cuneiform data:")
print(np.max(np.abs(data_cc[:,7,0]-data_th[:,14])))
print(np.max(np.abs(data_cc[:,7,1]-data_th[:,15])))

print(np.max(np.abs(data_cc[:,8,0]-data_th[:,16])))
print(np.max(np.abs(data_cc[:,8,1]-data_th[:,17])))

print(np.max(np.abs(data_cc[:,9,0]-data_th[:,18])))
print(np.max(np.abs(data_cc[:,9,1]-data_th[:,19])))
