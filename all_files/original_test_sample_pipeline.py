import numpy as np
import pickle
from graspsampler.GraspSampler import GraspSampler
import time
import json
import os

# define grasp sampler

grasp_save_dir = 'test_folder/grasps'
pc_save_dir = '/'
data_dir = "test_folder/training_data"

NUM_GRASPS = 1000000
NUM_LOCAL_GRASPS = 500
NUM_PCS = 10000
TARGET_PC_SIZE  = 1024



def sample_grasps(filename,path,name,scale,path_grasps,path_pcs):
    graspsampler = GraspSampler(seed=10)
    obj_filename = f'{path}'
    # load object
    graspsampler.update_object(obj_filename=obj_filename, name=name, obj_scale=scale)
    obj_pose_relative = graspsampler.obj.get_obj_mesh_mean()
    # sample grasps:
    _, normals, transforms, origins, quaternions,\
        alpha, _, collisions, qualities_1, qualities_2 = graspsampler.sample_grasps(number_of_grasps=NUM_GRASPS,
                                                                                 alpha_lim = [0, 2*np.pi],
                                                                                 beta_lim = [0, 0],
                                                                                 gamma_lim = [0, 0],
                                                                                 silent=False)
    # explore locally
    promising_candidates = (qualities_1 >0) & (qualities_2 >0)
    number_of_promising_candts = len(transforms[promising_candidates])

    if number_of_promising_candts != 0:
        new_transforms, new_origins, new_quaternions, new_collisions, new_qualities_1, new_qualities_2 = graspsampler.perturb_grasp_locally(number_of_grasps=NUM_LOCAL_GRASPS,
                                                                                normals=normals[promising_candidates],
                                                                                origins = origins[promising_candidates],
                                                                                alpha=alpha[promising_candidates],
                                                                                alpha_range=np.pi/36,
                                                                                beta_range=np.pi/18,
                                                                                gamma_range=np.pi/18,
                                                                                t_range=0.005,
                                                                                silent=False)
        # new_qualities_1 
        # new_qualities_2

        transforms = np.concatenate([transforms, new_transforms])
        origins = np.concatenate([origins, new_origins])
        quaternions = np.concatenate([quaternions, new_quaternions])
        collisions = np.concatenate([collisions, new_collisions])
        qualities_1 = np.concatenate([qualities_1, new_qualities_1])
        qualities_2 = np.concatenate([qualities_2, new_qualities_2])

        del new_transforms, new_origins, new_quaternions, new_collisions, new_qualities_1, new_qualities_2

    data_grasp = {}
    data_grasp['fname'] = obj_filename
    data_grasp['transforms'] = transforms
    data_grasp['translations'] = origins
    data_grasp['quaternions'] = quaternions
    data_grasp['collisions'] = collisions
    data_grasp['qualities_1'] = qualities_1
    data_grasp['qualities_2'] = qualities_2
    data_grasp['obj_pose_relative'] = obj_pose_relative
    data_grasp['scale'] =scale
    # add some labels for now only
    promising_candidates = (qualities_1 >0) & (qualities_2 >0)
    promising_candidates_count = len(transforms[promising_candidates])

    total_grasps = len(data_grasp['qualities_1'])

    with open(f'{path_grasps}/{filename}_original.pkl', 'wb') as fp:
        pickle.dump(data_grasp, fp, protocol=pickle.HIGHEST_PROTOCOL)
    del data_grasp, scale, normals, transforms, origins, alpha, collisions, qualities_1, qualities_2

    # # sample point clouds
    # pcs = graspsampler.get_multiple_random_pcs(number_of_pcs=NUM_PCS, 
    #                                         target_pc_size=TARGET_PC_SIZE, 
    #                                         depth_noise=0.01, 
    #                                         dropout=0.01)

    # with open(f'{path_pcs}/{filename}.pkl', 'wb') as fp:
    #     pickle.dump(pcs, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # del pcs
    promising_grasps_rate = promising_candidates_count/total_grasps
    print(f'Initial number of promising candidates: {number_of_promising_candts}')
    print(f'New number of promising candidates: {promising_candidates_count}')
    print(f'Total number of grasps gained: {total_grasps}')
    print(f"Promising Grasps Rate: {promising_grasps_rate}")
    print(f"done wiht {name}")

for object in os.listdir(data_dir):
    print(object)
    path_grasps = os.path.join(grasp_save_dir,object)
    path_pcs = os.path.join(pc_save_dir,object)
    for sample in os.listdir(f'{data_dir}/{object}'):
        filename = f'{data_dir}/{object}/{sample}'
        with open(filename) as json_file:
            data = json.load(json_file)
        name = data['id']
        path = data['path']
        scale = data["scale"]
        target_filename = sample.split(".")[0]
        sample_grasps(target_filename,path,name,scale,path_grasps,path_pcs)
