import numpy as np
import pickle

import trimesh
from graspsampler.GraspSampler import GraspSampler
import time
import json
import os
import matplotlib.pyplot as plt
from graspsampler.pointcloud_utils import *

# define grasp sampler

# grasp_save_dir = 'gitignore_stuff/grasps'
pc_save_dir = '/home/user/GRASP/grasp_network_semantic/data/pcs'
data_dir = "grasp_data/info"

NUM_PCS = 1000
TARGET_PC_SIZE  = None

graspsampler = GraspSampler(seed=10)

# idx = 12
# cat = 'bowl'



# camera = trimesh.creation.cone(radius=0.05, height=0.05)

# for i in range(NUM_PCS):

#     noisy_depth_map = add_noise_to_depth(depth_maps[i])
#     noisy_pc = depth_to_pointcloud(noisy_depth_map)
    
#     print(transferred_poses[i])
#     fig, ax = plt.subplots(2, figsize=(10,10))
#     ax[0].imshow(depth_maps[i])
#     ax[1].imshow(noisy_depth_map)
#     plt.show()

#     scene = trimesh.Scene()
#     # scene.add_geometry(graspsampler.obj.mesh)
#     scene.add_geometry(trimesh.points.PointCloud(pcs[i]))
#     n_pc = trimesh.points.PointCloud(noisy_pc)
#     n_pc.colors = [255,0,0,255]
#     n_pc.apply_transform(np.linalg.inv(transferred_poses[i]))
#     scene.add_geometry(n_pc)
#     # camera.apply_transform(camera_pose[0])
#     # scene.add_geometry(camera)
#     scene.show()





def process(idx):
    # cat = 'mug'

    name = f"{cat}{idx:03}" 
    print(name)

    ext = 'obj'
    if cat in ['box', 'cylinder']:
        ext = 'stl'

    path = f"grasp_data/meshes/{cat}/{cat}{idx:03}.{ext}"
    metadata_filename = f"grasp_data/info/{cat}/{cat}{idx:03}.json"

    try:
        metadata = json.load(open(metadata_filename,'r'))
    except:
        return

    scale = metadata["scale"]


    # load object
    graspsampler.update_object(obj_filename=path, name=name, obj_scale=scale)
    obj_pose_relative = graspsampler.obj.get_obj_mesh_mean()

    # sample point clouds
    pcs, depth_maps, transferred_poses = graspsampler.get_multiple_random_pcs(number_of_pcs=NUM_PCS, 
                                        target_pc_size=TARGET_PC_SIZE, 
                                        depth_noise=[0,0], 
                                        dropout=0)

    camera_poses = np.linalg.inv(transferred_poses)

    data = {}
    data['obj_pose_relative'] = obj_pose_relative
    data['pcs'] = pcs
    data['depth_maps'] = depth_maps
    data['camera_poses'] = camera_poses
    
    with open(f'{pc_save_dir}/{cat}/{name}.pkl', 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    del data



    

# def do(cat):
#     for i in range(0,22):
#         process(cat, i)


from joblib import Parallel, delayed
cats = ["fork","hammer","spatula","pan","scissor"]
# for cat in cats:
for cat in cats:
    Parallel(n_jobs=20)(delayed(process)(i) for i in range(20))


########## LATER LOOK AT #####################
# print(np.expand_dims(camera_pose[:3,3], 0).shape, pcs[0].shape)
# d = np.linalg.norm(pcs[0] - np.expand_dims(camera_pose[:3,3], 0), axis=1)
# top_d_ind = np.argsort(d)[-30:]
# top_d = pcs[0][top_d_ind]


# print(top_d)
# plane = trimesh.points.plane_fit(top_d)
# print(plane)
# # scene.add_geometry(trimesh.points.PointCloud(plane))

# from scipy.spatial.transform import Rotation as R
# plane_pose = np.eye(4)
# plane_pose[:3,:3] = R.from_rotvec(plane[1]).as_matrix()
# plane_pose[:3,3] = plane[0]

# scene.add_geometry(trimesh.primitives.Box(extents=[1,1,0.01], transform=plane_pose))
# scene.show()
