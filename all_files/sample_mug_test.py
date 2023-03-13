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
from graspsampler.utils import gripper_bd

trial = 100
category = 'mug'
# i=0
NUM_GRASPS = 500
NUM_LOCAL_GRASPS = 10
NUM_FAR_GRASPS = 10

def process(i):

    save_dir = 'grasp_data_generated'
    obj_filename = f'grasp_data/meshes/{category}/{category}{i:03}.obj'    
    metadata_filename = f'grasp_data/info/{category}/{category}{i:03}.json'
    metadata = json.load(open(metadata_filename,'r'))

    # define grasp sampler
    graspsampler = GraspSampler(seed=trial)

    # load object
    graspsampler.update_object(obj_filename=obj_filename, name=obj_filename, obj_scale=metadata['scale'])
    obj_pose_relative = graspsampler.obj.get_obj_mesh_mean()

    start_time = time.time()
    # general uniform sampling

    _, normals, transforms, origins, quaternions,\
        alpha, beta, gamma, _, qualities = graspsampler.sample_grasps_mug(number_of_grasps=NUM_GRASPS,
                                                                    alpha_lim = [np.pi/2, np.pi/2],
                                                                    beta_lim = [0, 0],
                                                                    gamma_lim = [0, 0],
                                                                    silent=True)
    is_promising = np.zeros_like(qualities)
    is_promising[qualities > 0] = 1

    initial_promising_candidate_count = np.sum(is_promising)
    candidate_indices = np.argwhere(is_promising == 1).squeeze()

    # add some bad grasps for candidate too
    random_candidates = np.argwhere(is_promising == 0).squeeze()
    random_candidates = np.random.choice(random_candidates, size=int(NUM_GRASPS*0.01))
    if len(random_candidates) == 0:
        candidate_indices = candidate_indices

    is_promising[candidate_indices] = 1


##### need to change file name here #######
    main_save_dir = f'{save_dir}/{category}/{category}{i:03}'
    Path(main_save_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f'../isaac/test_mug{i:03}.npz',
                        transforms = transforms,
                        quaternions=quaternions,
                        translations=origins,
                        is_promising=is_promising,
                        obj_pose_relative=obj_pose_relative)

    

    # # local sampling
    # total_promising_candidate_count = initial_promising_candidate_count
    # total_grasps = NUM_GRASPS
    # for k, idx in enumerate(candidate_indices):
    #     idx = [idx]
    #     new_transforms, new_origins, new_quaternions, new_qualities = graspsampler.perturb_grasp_locally(number_of_grasps=NUM_LOCAL_GRASPS,
    #                                                                         normals=normals[idx],
    #                                                                         origins = origins[idx],
    #                                                                         alpha=alpha[idx],
    #                                                                         beta=beta[idx],
    #                                                                         gamma=gamma[idx],
    #                                                                         alpha_range=np.pi/18/0.5,
    #                                                                         beta_range=np.pi/12/0.5,
    #                                                                         gamma_range=np.pi/12/0.5,
    #                                                                         t_range=0.05,
    #                                                                         silent=True)
    #     _is_promising = np.zeros_like(new_qualities)
    #     _is_promising[new_qualities > 0] = 1


    #     np.savez_compressed(f'{main_save_dir}/main{trial}_{int(idx[0]):08}.npz',
    #                     transforms = new_transforms,
    #                     quaternions=new_quaternions,
    #                     translations=new_origins,
    #                     is_promising=_is_promising,
    #                     obj_pose_relative=obj_pose_relative)

    #     total_promising_candidate_count += np.sum(_is_promising)
    #     total_grasps += NUM_LOCAL_GRASPS

    #     del new_transforms, new_origins, new_quaternions, new_qualities, _is_promising


    ## sample far grasps
    # graspsampler2 = GraspSampler(seed=trial)
    # if isinstance(metadata['scale'], list):
    #     far_scale = [metadata['scale'][0]*2.5,metadata['scale'][1]*2.5,metadata['scale'][2]*2.5]
    # else:
    #     far_scale = metadata['scale']*2.5
    # graspsampler2.update_object(obj_filename=obj_filename, name=obj_filename, obj_scale=far_scale)
    # _, _, transforms2, origins2, quaternions2,\
    #     _, _, _, _, qualities2 = graspsampler2.sample_grasps(number_of_grasps=NUM_FAR_GRASPS,
    #                                                                 alpha_lim = [0, 2*np.pi],
    #                                                                 beta_lim = [-np.pi, np.pi],
    #                                                                 gamma_lim = [0, 2*np.pi],
    #                                                                 silent=True)

    # np.savez_compressed(f'{main_save_dir}/main{trial}_far_grasps.npz',
    #                 transforms = transforms2,
    #                 quaternions=quaternions2,
    #                 translations=origins2,
    #                 is_promising=np.zeros_like(qualities2),
    #                 obj_pose_relative=obj_pose_relative)

    # del transforms2, origins2, quaternions2, qualities2, graspsampler2

    # print('---------------------------------------------')   
    print(f'Inital candidates are {int(initial_promising_candidate_count)} out of {NUM_GRASPS}: {initial_promising_candidate_count/NUM_GRASPS*100:.2f} %.')
    # print(f'New candidates are {int(total_promising_candidate_count)} out of {total_grasps}: {total_promising_candidate_count/total_grasps*100:.2f} %.')
    # print(f'Done for {category}{i:03} [trial {trial}] in {time.time()-start_time} seconds.')

    # del _, normals, transforms, origins, quaternions, alpha, beta, gamma, qualities, is_promising, graspsampler


######## need to modify the code here #############
from joblib import Parallel, delayed
Parallel(n_jobs=21)(delayed(process)(i) for i in range(21))
