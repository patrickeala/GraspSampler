import numpy as np
from pathlib import Path

#from trimesh.permutate import transform
from graspsampler.common import PandaGripper, Scene, Object
import matplotlib.pyplot as plt
import trimesh.transformations as tra
import trimesh
import pickle

from graspsampler.GraspSampler import GraspSampler

# define grasp sampler
graspsampler = GraspSampler(seed=10)
grasp_save_dir = 'sample_grasp_dataset/grasps'
pc_save_dir = 'sample_grasp_dataset/pcs'

NUM_GRASPS = 20000

SIZE = 5000

import time

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

def process(i):
    start_t = time.time()
    obj_filename = f'assets/meshes_10_objects/{objs[i]}/google_16k/textured.obj' 
    
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

    pc, _ = graspsampler.get_pc(full_pc=True)

    data_grasp = {}
    data_grasp['fname'] = objs[i]
    data_grasp['scale'] = 1
    data_grasp['full_pc'] = pc
    data_grasp['transforms'] = transforms
    data_grasp['translations'] = origins
    data_grasp['quaternions'] = quaternions
    data_grasp['collisions'] = collisions
    data_grasp['qualities_1'] = qualities_1
    data_grasp['qualities_2'] = qualities_2
    data_grasp['obj_pose_relative'] = obj_pose_relative

    with open(f'isaac_test_10_meshes/{objs[i]}.pkl', 'wb') as fp:
        pickle.dump(data_grasp, fp, protocol=pickle.HIGHEST_PROTOCOL)


    # # add some labels for now only
    # sorted1_idx = np.argsort(qualities_1)
    # sorted2_idx = np.argsort(qualities_2)
    # top1_idx = sorted1_idx[-SIZE:]
    # top2_idx = sorted2_idx[-SIZE:]
    # bottom1_idx = sorted1_idx[:SIZE]
    # bottom2_idx = sorted2_idx[:SIZE]

    # data_grasp = {}
    # data_grasp['fname'] = objs[i]
    # data_grasp['scale'] = 1
    # data_grasp['full_pc'] = pc
    # data_grasp['transforms'] = transforms[top1_idx]
    # data_grasp['translations'] = origins[top1_idx]
    # data_grasp['quaternions'] = quaternions[top1_idx]
    # data_grasp['collisions'] = collisions[top1_idx]
    # data_grasp['qualities_1'] = qualities_1[top1_idx]
    # data_grasp['qualities_2'] = qualities_2[top1_idx]
    # data_grasp['obj_pose_relative'] = obj_pose_relative
    # with open(f'isaac_test_10_meshes/{objs[i]}_top1.pkl', 'wb') as fp:
    #     pickle.dump(data_grasp, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # # TOP 2
    
    # data_grasp = {}
    # data_grasp['fname'] = objs[i]
    # data_grasp['scale'] = 1
    # data_grasp['full_pc'] = pc
    # data_grasp['transforms'] = transforms[top2_idx]
    # data_grasp['translations'] = origins[top2_idx]
    # data_grasp['quaternions'] = quaternions[top2_idx]
    # data_grasp['collisions'] = collisions[top2_idx]
    # data_grasp['qualities_1'] = qualities_1[top2_idx]
    # data_grasp['qualities_2'] = qualities_2[top2_idx]
    # data_grasp['obj_pose_relative'] = obj_pose_relative

    # with open(f'isaac_test_10_meshes/{objs[i]}_top2.pkl', 'wb') as fp:
    #     pickle.dump(data_grasp, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # # BOTTOM 1
    
    # data_grasp = {}
    # data_grasp['fname'] = objs[i]
    # data_grasp['scale'] = 1
    # data_grasp['full_pc'] = pc
    # data_grasp['transforms'] = transforms[bottom1_idx]
    # data_grasp['translations'] = origins[bottom1_idx]
    # data_grasp['quaternions'] = quaternions[bottom1_idx]
    # data_grasp['collisions'] = collisions[bottom1_idx]
    # data_grasp['qualities_1'] = qualities_1[bottom1_idx]
    # data_grasp['qualities_2'] = qualities_2[bottom1_idx]
    # data_grasp['obj_pose_relative'] = obj_pose_relative

    # with open(f'isaac_test_10_meshes/{objs[i]}_bottom1.pkl', 'wb') as fp:
    #     pickle.dump(data_grasp, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # # BOTTOM 2
    
    # data_grasp = {}
    # data_grasp['fname'] = objs[i]
    # data_grasp['scale'] = 1
    # data_grasp['full_pc'] = pc
    # data_grasp['transforms'] = transforms[bottom2_idx]
    # data_grasp['translations'] = origins[bottom2_idx]
    # data_grasp['quaternions'] = quaternions[bottom2_idx]
    # data_grasp['collisions'] = collisions[bottom2_idx]
    # data_grasp['qualities_1'] = qualities_1[bottom2_idx]
    # data_grasp['qualities_2'] = qualities_2[bottom2_idx]
    # data_grasp['obj_pose_relative'] = obj_pose_relative

    # with open(f'isaac_test_10_meshes/{objs[i]}_bottom2.pkl', 'wb') as fp:
    #     pickle.dump(data_grasp, fp, protocol=pickle.HIGHEST_PROTOCOL)


    # del data_grasp, normals, transforms, origins, alpha, collisions, qualities_1, qualities_2



    print(f'Stats for object {i}:')
    print('Time taken [seconds]: ', time.time()-start_t)
    print('')

from joblib import Parallel, delayed
Parallel(n_jobs=10)(delayed(process)(i) for i in range(10))
# process(10)
