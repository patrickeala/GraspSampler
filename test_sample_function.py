from sample_grasps_for_pcl import sample_grasps_for_pcls
import pickle
import numpy as np

# object setup
cat = 'bottle'
idx = 0

# get pcls
_pcls = f"grasp_data/pcls/{cat}/{cat}{idx:03}.pkl"
pcls = pickle.load(open(_pcls,"rb"))

# get grasps
quaternions, translations= sample_grasps_for_pcls(pcls, nun_of_grasps=1)
print("input shape: ", pcls.shape)
print("output shape: ", quaternions.shape, translations.shape)


np.savez_compressed(f'{cat}_{idx}.npz',
                    quaternions=quaternions,
                    translations=translations,
                    )
