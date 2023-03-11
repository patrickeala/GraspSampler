from typing import get_args
import numpy as np
import time
#from trimesh.permutate import transform
import trimesh
import pickle
from graspsampler.GraspSampler import GraspSampler
from graspsampler.common import Scene, Object, PandaGripper
import json
from pathlib import Path

i = 6
trial = 2
NUM_GRASPS = 100000
NUM_LOCAL_GRASPS = 50

objs = {
    0: '004_sugar_box',
    1: '005_tomato_soup_can',
    2: '006_mustard_bottle',
    3: '007_tuna_fish_can',
    4: '010_potted_meat_can',
    5: '014_lemon',
    6: '025_mug',
    7: '026_sponge',
    8: '061_foam_brick',
    9: '065_j_cups',
    10: '001_chips_can'
}

obj_filename = f'assets/meshes_10_objects/{objs[i]}/google_16k/textured.obj'
save_dir = 'test_data2/test_data_grasps/'

# define grasp sampler  
graspsampler = GraspSampler(seed=trial)
NUM_PCS = 1000
TARGET_PC_SIZE  = 1024
# load object
graspsampler.update_object(obj_filename=obj_filename, name=obj_filename, obj_scale=1)
obj_pose_relative = graspsampler.obj.get_obj_mesh_mean()


start_time = time.time()
# general uniform sampling

_, normals, transforms, origins, quaternions,\
    alpha, beta, gamma, _, qualities = graspsampler.sample_grasps(number_of_grasps=NUM_GRASPS,
                                                                alpha_lim = [0, 2*np.pi],
                                                                beta_lim = [-np.pi, np.pi],
                                                                gamma_lim = [0, 2*np.pi],
                                                                silent=True)
is_promising = np.zeros_like(qualities)
is_promising[qualities > 0] = 1

main_save_dir = f'{save_dir}/{objs[i]}'
Path(main_save_dir).mkdir(parents=True, exist_ok=True)
np.savez_compressed(f'{main_save_dir}/main{trial}.npz',
                    transforms = transforms,
                    quaternions=quaternions,
                    translations=origins,
                    is_promising=is_promising,
                    obj_pose_relative=obj_pose_relative)

initial_promising_candidate_count = np.sum(is_promising)
candidate_indices = np.argwhere(is_promising == 1)

# local sampling
total_promising_candidate_count = initial_promising_candidate_count
total_grasps = NUM_GRASPS
for k, idx in enumerate(candidate_indices):
    new_transforms, new_origins, new_quaternions, new_qualities = graspsampler.perturb_grasp_locally(number_of_grasps=NUM_LOCAL_GRASPS,
                                                                        normals=normals[idx],
                                                                        origins = origins[idx],
                                                                        alpha=alpha[idx],
                                                                        beta=beta[idx],
                                                                        gamma=gamma[idx],
                                                                        alpha_range=np.pi/18/2,
                                                                        beta_range=np.pi/12/2,
                                                                        gamma_range=np.pi/12/2,
                                                                        t_range=0.05/2,
                                                                        silent=True)
    _is_promising = np.zeros_like(new_qualities)
    _is_promising[new_qualities > 0] = 1

    np.savez_compressed(f'{main_save_dir}/main{trial}_{int(idx):08}.npz',
                    transforms = new_transforms,
                    quaternions=new_quaternions,
                    translations=new_origins,
                    is_promising=_is_promising,
                    obj_pose_relative=obj_pose_relative)

    total_promising_candidate_count += np.sum(_is_promising)
    total_grasps += NUM_LOCAL_GRASPS

    del new_transforms, new_origins, new_quaternions, new_qualities, _is_promising

print('---------------------------------------------')    
print(f'Inital candidates are {int(initial_promising_candidate_count)} out of {NUM_GRASPS}: {initial_promising_candidate_count/NUM_GRASPS*100:.2f} %.')
print(f'New candidates are {int(total_promising_candidate_count)} out of {total_grasps}: {total_promising_candidate_count/total_grasps*100:.2f} %.')


# sample point clouds

# pcs = graspsampler.get_multiple_random_pcs(number_of_pcs=NUM_PCS, 
#                                         target_pc_size=TARGET_PC_SIZE, 
#                                         depth_noise=0, 
#                                         dropout=0)

# with open(f'test_data_pcls/{object}.pkl', 'wb') as fp:
#     pickle.dump(pcs, fp, protocol=pickle.HIGHEST_PROTOCOL)
# del pcs


