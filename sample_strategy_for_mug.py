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




category = 'bowl'
trial = 6
NUM_GRASPS = 100000#00
NUM_LOCAL_GRASPS = 5000#0

def process(i):
    obj_filename = f'grasp_data/meshes/{category}/{category}{i:03}.obj'
    metadata_filename = f'grasp_data/info/{category}/{category}{i:03}.json'
    metadata = json.load(open(metadata_filename,'r'))

    save_dir = 'grasp_data_generated'

    # define grasp sampler
    graspsampler = GraspSampler(seed=trial)

    # load object
    graspsampler.update_object(obj_filename=obj_filename, name=obj_filename, obj_scale=metadata['scale'])
    obj_pose_relative = graspsampler.obj.get_obj_mesh_mean()


    start_time = time.time()
    # general uniform sampling

    _, normals, transforms, origins, quaternions,\
        alpha, beta, gamma, _, qualities = graspsampler.sample_grasps(number_of_grasps=NUM_GRASPS,
                                                                    alpha_lim = [0, 2*np.pi],
                                                                    beta_lim = [-np.pi/2, np.pi/2],
                                                                    gamma_lim = [0, np.pi],
                                                                    silent=True)
    is_promising = np.zeros_like(qualities)
    is_promising[qualities > 0] = 1

    main_save_dir = f'{save_dir}/{category}/{category}{i:03}'
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
                                                                            alpha_range=np.pi/18/1.5,
                                                                            beta_range=np.pi/12/1.5,
                                                                            gamma_range=np.pi/12/1.5,
                                                                            t_range=0.05/1.5,
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
    print(f'Done for {category}{i:03} [trial {trial}] in {time.time()-start_time} seconds.')

    del _, normals, transforms, origins, quaternions, alpha, beta, gamma, qualities, is_promising


from joblib import Parallel, delayed
Parallel(n_jobs=20)(delayed(process)(i) for i in range(20))
