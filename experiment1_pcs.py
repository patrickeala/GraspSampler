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


NUM_PCS = 1000
TARGET_PC_SIZE  = None
i = 0

objs = {
    0: '004_sugar_box',
    1: '005_tomato_soup_can',
    4: '010_potted_meat_can',
    6: '025_mug',
    7: '026_sponge',
    8: '061_foam_brick',
    9: '065_j_cups',
    11: '024_bowl'
}

for i in objs.keys():
    # define grasp sampler  
    if i != 8:
        continue
    graspsampler = GraspSampler(seed=10)

    # load object
    obj_filename = f'assets/meshes_10_objects/{objs[i]}/google_16k/textured.obj'
    graspsampler.update_object(obj_filename=obj_filename, name=obj_filename, obj_scale=1)
    obj_pose_relative = graspsampler.obj.get_obj_mesh_mean()

    print(obj_filename)

    pcs = graspsampler.get_multiple_random_pcs(number_of_pcs=NUM_PCS, 
                                            target_pc_size=TARGET_PC_SIZE, 
                                            depth_noise=0, 
                                            dropout=0)


    with open(f'experiment1_pcs/{objs[i]}.pkl', 'wb') as fp:
        pickle.dump([pcs, obj_pose_relative], fp, protocol=pickle.HIGHEST_PROTOCOL)
    del pcs, graspsampler