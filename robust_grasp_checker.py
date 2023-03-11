import pickle
import numpy as np
from numpy.lib.function_base import diff
from tqdm.auto import tqdm
from scipy.spatial.transform import Rotation as Rot
from tqdm.auto import tqdm

# set thresholds
d_t_hat = 0.001 # meters
d_alpha_hat = 1 # degrees
d_alpha_hat *= np.pi/180.0 # degrees

# load grasps
i=0
fname=f'sample_grasp_dataset/box{i:03}.pkl'
data = pickle.load(open(fname, 'rb'))
data['transforms'] = data['transforms'][:100000]
data['qualities_1'] = data['qualities_1'][:100000]
data['qualities_2'] = data['qualities_2'][:100000]

qual1 = np.array( data['qualities_1'] )
qual2 = np.array( data['qualities_2'])
mask1 = qual1 >= 0.5
mask2 = qual2 >= 0.5
positive_transforms = data['transforms'][mask1 & mask2]
negative_transforms = data['transforms'][~(mask1 & mask2)]
#pos_grasps = np.random.choice(len(masked_transforms), size=10)


print(f"Total grasp: {len(data['transforms'])}")
print(f"Positive grasp: {len(positive_transforms)}")
print(f"Negative grasp: {len(negative_transforms)}")


# t_prime = negative_transforms[:,:3,3]
# R_prime = negative_transforms[:,:3,:3]
# robust_labels = np.zeros(len(data['transforms']))
# for i, pos_grasp in enumerate(positive_transforms):
#     t = pos_grasp[:3,3]
#     d_t = np.linalg.norm(t - t_prime, axis=1)
#     if len(d_t[d_t <= d_t_hat]) == 0:
#         # convert transform to
#         print('here')
#         R = pos_grasp[:3,:3]
#         #print('R')
#         #print(R)
#         #print('RT')
#         #print(R_prime)
#         #print('R . RT:')
#         #print(R_prime@R.T)
#         _tr_R = np.trace(R_prime@R.T, axis1=1, axis2=2)
#         for l, _ in enumerate(_tr_R):
#             if (_ <= -1.0) or (_ >= 1.0):
#                 print(_)
#                 print('weird matrix')
#                 print(R_prime[l])
#         # if len(_tr_R[(_tr_R >= -1.0) & (_tr_R <= 1.0)]) > 0:
#         #     print('WARNING', _tr_R)
#         d_alpha = 0.5*np.arccos( _tr_R ) - 0.5
        
#         if len(d_alpha[d_alpha <= d_alpha_hat]) == 0:
#             robust_labels[i] = 1.0


t_prime = negative_transforms[:,:3, 3]
q_prime = Rot.from_matrix(negative_transforms[:,:3,:3]).as_quat()
robust_labels = np.zeros(len(data['transforms']))

t = positive_transforms[:,:3,3]



for i, pos_grasp in tqdm(enumerate(positive_transforms), total=len(positive_transforms)):
    t = pos_grasp[:3,3]
    d_t = np.linalg.norm(t - t_prime, axis=1)
    if len(d_t[d_t <= d_t_hat]) == 0:
        # convert transform to
        #print('here')
        q = Rot.from_matrix(pos_grasp[:3,:3]).as_quat()
        q_q_prime = np.abs( np.sum(np.multiply(q, q_prime), axis=1) )
        d_alpha = 2*np.arccos(q_q_prime)
        #print(q_q_prime.shape)       
        # for l, _ in enumerate(_tr_R):
        #     if (_ <= -1.0) or (_ >= 1.0):
        #         print(_)
        #         print('weird matrix')
        #         print(R_prime[l])
        # # if len(_tr_R[(_tr_R >= -1.0) & (_tr_R <= 1.0)]) > 0:
        # #     print('WARNING', _tr_R)
        # d_alpha = 0.5*np.arccos( _tr_R ) - 0.5
        
        if len(d_alpha[d_alpha <= d_alpha_hat]) == 0:
            robust_labels[i] = 1.0


print(np.sum(robust_labels))