import numpy as np

#from trimesh.permutate import transform
import pickle
from graspsampler.GraspSampler import GraspSampler

# for handler in logging.root.handlers[:]:
#     logging.root.removeHandler(handler)
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("log_file.log"),
#         logging.StreamHandler()
#     ]
# )
# log = logging.getLogger(__name__)

# print(log.handlers)

# define grasp sampler
graspsampler = GraspSampler(seed=10)
grasp_save_dir = 'temp_grasps'
pc_save_dir = 'sample_grasp_dataset/pcs'

NUM_GRASPS = 15
NUM_LOCAL_GRASPS = 5

NUM_PCS = 10000
TARGET_PC_SIZE  = 1024

import time


def process(i):
    start_t = time.time()
    obj_filename = f'assets/sample_files/box{i:03}.stl'
    
    # load object
    graspsampler.update_object(obj_filename=obj_filename, name=obj_filename)
    obj_pose_relative = graspsampler.obj.get_obj_mesh_mean()

    # sample grasps:
    _, normals, transforms, origins, quaternions,\
        alpha, _, collisions, qualities_1, qualities_2 = graspsampler.sample_grasps(number_of_grasps=NUM_GRASPS,
                                                                                 alpha_lim = [0, 2*np.pi],
                                                                                 beta_lim = [0, 0],
                                                                                 gamma_lim = [0, 0],
                                                                                 silent=True)
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
                                                                                silent=True)

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


    for kkk in range(len(data_grasp['transforms'])):
        print(f'-------{kkk}------------')
        print('Transform:')
        print(data_grasp['transforms'][kkk])
        print('Translation:')
        print(data_grasp['translations'][kkk])

    # add some labels for now only
    promising_candidates = (qualities_1 >0) & (qualities_2 >0)
    promising_candidates_count = len(transforms[promising_candidates])

    with open(f'{grasp_save_dir}/box{i:03}.pkl', 'wb') as fp:
        pickle.dump(data_grasp, fp, protocol=pickle.HIGHEST_PROTOCOL)
    del data_grasp, normals, transforms, origins, alpha, collisions, qualities_1, qualities_2

    # # sample point clouds
    # pcs = graspsampler.get_multiple_random_pcs(number_of_pcs=NUM_PCS, 
    #                                         target_pc_size=TARGET_PC_SIZE, 
    #                                         depth_noise=0.01, 
    #                                         dropout=0.01)

    # with open(f'{pc_save_dir}/box{i:03}.pkl', 'wb') as fp:
    #     pickle.dump(pcs, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # del pcs


    print(f'Stats for object {i}:')
    print(f'Initial number of promising candidates: {number_of_promising_candts}')
    print(f'New number of promising candidates: {promising_candidates_count}')
    print(f'Time taken [seconds]:  {time.time()-start_t}')



# from joblib import Parallel, delayed
# Parallel(n_jobs=20)(delayed(process)(i) for i in range(20))
process(1)
