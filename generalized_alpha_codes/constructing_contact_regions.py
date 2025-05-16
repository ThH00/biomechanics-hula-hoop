import numpy as np
from itertools import product
import os

# [B[k], C[k]] if A[k]=1
possible_normal_config_per_contact = np.array([[0,0],[1,0],[1,1]])

# [D[k], E[k]] if A[k]=1
possible_friction_config_per_contact = np.array([[0,0],[1,0],[1,1]])

possible_config_per_contact = np.array([np.hstack((normal_contact, friction_contact)) for normal_contact, friction_contact in product(possible_normal_config_per_contact, possible_friction_config_per_contact)])

A = np.array([1,0])

# if A == np.array([0,0]):
unique_contacts_a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# if A == np.array([1,0]):
unique_contacts_b = np.zeros((9,10))
unique_contacts_b[:,0] = np.ones(9)
unique_contacts_b[:,[2,4,6,8]] = possible_config_per_contact
# if A == np.array([0,1]):
unique_contacts_c = np.zeros((9,10))
unique_contacts_c[:,1] = np.ones(9)
unique_contacts_c[:,[3,5,7,9]] = possible_config_per_contact
# if A == np.array([1,1]):
# [B, C] if A = [1,1]
possible_normal_config = np.array([[0,0,0,0],
                                    [1,0,0,0],
                                    [1,0,1,0],
                                    [0,1,0,0],
                                    [0,1,0,1],
                                    [1,1,0,0],
                                    [1,1,1,0],
                                    [1,1,0,1],
                                    [1,1,1,1]])
# [D, E] if A = [1,1]
possible_friction_config = np.array([[0,0,0,0],
                                    [1,0,0,0],
                                    [1,0,1,0],
                                    [0,1,0,0],
                                    [0,1,0,1],
                                    [1,1,0,0],
                                    [1,1,1,0],
                                    [1,1,0,1],
                                    [1,1,1,1]])
unique_contacts_d = np.ones((81,10))
unique_contacts_d[:,2:10] = np.array([np.hstack((normal_contact, friction_contact)) for normal_contact, friction_contact in product(possible_normal_config, possible_friction_config)])

output_path = os.path.join(os.getcwd(), "unique_contacts")
os.makedirs(output_path, exist_ok=True)
np.save(f'{output_path}/unique_contacts_a.npy', unique_contacts_a)
np.save(f'{output_path}/unique_contacts_b.npy', unique_contacts_b)
np.save(f'{output_path}/unique_contacts_c.npy', unique_contacts_c)
np.save(f'{output_path}/unique_contacts_d.npy', unique_contacts_d)



print('done')