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
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-trial", "--trial", help="index of trial")
parser.add_argument("-cat", "--cat", help="category of object")
args = parser.parse_args()


trial = int(args.trial)
category = args.cat
# i=0
NUM_GRASPS = 15000
NUM_LOCAL_GRASPS = 100
NUM_FAR_GRASPS = 1000

def process(i):
    print(f"getting trial {trial} for {category}")
    save_dir = 'grasp_data_generated'
    obj_filename = f'grasp_data/meshes/{category}/{category}{i:03}.obj'
    metadata_filename = f'grasp_data/info/{category}/{category}{i:03}.json'
    metadata = json.load(open(metadata_filename,'r'))

    # define grasp sampler
    graspsampler = GraspSampler(seed=trial)
    
    # define data directory
    main_save_dir = f'{save_dir}/{category}/{category}{i:03}'
    Path(main_save_dir).mkdir(parents=True, exist_ok=True)


    # load object
    graspsampler.update_object(obj_filename=obj_filename, name=obj_filename, obj_scale=metadata['scale'])
    obj_pose_relative = graspsampler.obj.get_obj_mesh_mean()

    start_time = time.time()
    # general uniform sampling

    _, normals, transforms, origins, quaternions,\
        alpha, beta, gamma, _, is_promising = graspsampler.sample_grasps(number_of_grasps=NUM_GRASPS,
                                                                    alpha_lim = [np.pi/2*((trial+1)%2), np.pi/2*((trial+1)%2)],
                                                                    beta_lim = [0, 0],
                                                                    gamma_lim = [0, 0],
                                                                    silent=False)

    initial_promising_candidate_count = np.sum(is_promising)
    candidate_indices = np.where(is_promising == 1)[0]


    np.savez_compressed(f'{main_save_dir}/main{trial}.npz',
                        transforms = transforms,
                        quaternions=quaternions,
                        translations=origins,
                        is_promising=is_promising,
                        obj_pose_relative=obj_pose_relative)

    

    # local sampling
    total_promising_candidate_count = initial_promising_candidate_count
    total_grasps = NUM_GRASPS
    for k, idx in enumerate(candidate_indices):
        idx = [idx]
        new_transforms, new_origins, new_quaternions, _is_promising = graspsampler.perturb_grasp_locally(number_of_grasps=NUM_LOCAL_GRASPS,
                                                                            normals=normals[idx],
                                                                            origins = origins[idx],
                                                                            alpha=alpha[idx],
                                                                            beta=beta[idx],
                                                                            gamma=gamma[idx],
                                                                            alpha_range=np.pi/18/0.5,
                                                                            beta_range=np.pi/12/0.5,
                                                                            gamma_range=np.pi/12/0.5,
                                                                            t_range=0.05/0.5,
                                                                            silent=False)


        np.savez_compressed(f'{main_save_dir}/main{trial}_{int(idx[0]):08}.npz',
                        transforms = new_transforms,
                        quaternions=new_quaternions,
                        translations=new_origins,
                        is_promising=_is_promising,
                        obj_pose_relative=obj_pose_relative)

        total_promising_candidate_count += np.sum(_is_promising)
        total_grasps += NUM_LOCAL_GRASPS

        del new_transforms, new_origins, new_quaternions, _is_promising


    ## sample far grasps
    graspsampler2 = GraspSampler(seed=trial)
    if isinstance(metadata['scale'], list):
        far_scale = [metadata['scale'][0]*2.5,metadata['scale'][1]*2.5,metadata['scale'][2]*2.5]
    else:
        far_scale = metadata['scale']*2.5
    graspsampler2.update_object(obj_filename=obj_filename, name=obj_filename, obj_scale=far_scale)
    _, _, transforms2, origins2, quaternions2,\
        _, _, _, _, is_promising2 = graspsampler2.sample_grasps(number_of_grasps=NUM_FAR_GRASPS,
                                                                    alpha_lim = [0, 2*np.pi],
                                                                    beta_lim = [-np.pi, np.pi],
                                                                    gamma_lim = [0, 2*np.pi],
                                                                    silent=True)

    np.savez_compressed(f'{main_save_dir}/main{trial}_far_grasps.npz',
                    transforms = transforms2,
                    quaternions=quaternions2,
                    translations=origins2,
                    is_promising=is_promising2,
                    obj_pose_relative=obj_pose_relative)

    del transforms2, origins2, quaternions2, graspsampler2

    print('---------------------------------------------')   
    print(f'Inital candidates are {int(initial_promising_candidate_count)} out of {NUM_GRASPS}: {initial_promising_candidate_count/NUM_GRASPS*100:.2f} %.')
    print(f'New candidates are {int(total_promising_candidate_count)} out of {total_grasps}: {total_promising_candidate_count/total_grasps*100:.2f} %.')
    print(f'Done for {category}{i:03} [trial {trial}] in {time.time()-start_time} seconds.')

    del _, normals, transforms, origins, quaternions, alpha, beta, gamma, is_promising, graspsampler


# transforms = np.concatenate(big_transforms)
# print(transforms.shape)

# scene = trimesh.Scene()
# graspsampler.obj.mesh.visual.face_colors = [255,0,0,60]
# scene.add_geometry(graspsampler.obj.mesh)
# for transform in transforms:
#     gripper = gripper_bd()
#     scene.add_geometry(gripper, transform=transform)

# scene.show()


from joblib import Parallel, delayed
# _indcs = [0, 8, 12, 17]
# process(12)
Parallel(n_jobs=10)(delayed(process)(i) for i in range(20))
