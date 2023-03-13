from gettext import translation
import numpy as np
import argparse
import trimesh
import pickle
from graspsampler.GraspSampler import GraspSampler
from graspsampler.common import PandaGripper, Scene, Object

from scipy.spatial.transform import Rotation as R
from graspsampler import utils
from graspsampler.common import PandaGripper, Scene

import sys
cat = sys.argv[-2]
# idx = int(sys.argv[-2])
split = sys.argv[-1]
trial = 1

dataset = 'fakeworld_grasp_data_table/preprocessed_table'
# dataset = 'fakeworld_grasp_data_table/preprocessed_table_boxcyl'
# dataset = 'fakeworld_grasp_data_friction/preprocessed_friction_boxcyl'
label_path = f"/home/crslab/GRASP/isaac/{dataset}/{cat}/isaac_labels_{split}.npy"
quat_path = f"/home/crslab/GRASP/isaac/{dataset}/{cat}/quaternions_{split}.npy"
trans_path = f"/home/crslab/GRASP/isaac/{dataset}/{cat}/translations_{split}.npy"
meta_path = f"/home/crslab/GRASP/isaac/{dataset}/{cat}/metadata_{split}.npy"

labels = np.load(label_path)
quaternions = np.load(quat_path)
translations = np.load(trans_path)
meta = np.load(meta_path)


print(f"Total grasps: {quaternions.shape[0]}")
print(f"Total grasps: {translations.shape[0]}")
print(f"Total grasps: {labels.shape[0]}")






def load_object_mesh(idx):
    # Get object mesh
    path_to_assets = 'grasp_data' # object mesh and json path
    obj_name = f'{cat}{idx:03}'


    path_to_obj_mesh = f'{path_to_assets}/meshes/{cat}/{obj_name}.stl'
    if (cat == 'mug') or (cat == "bottle") or (cat == "bowl") or (cat == "hammer") or (cat == "scissor") or (cat == "fork"):
        path_to_obj_mesh = f'{path_to_assets}/meshes/{cat}/{obj_name}.obj'
    # scale = utils.get_scale(f'{path_to_assets}/info/{cat}/{obj_name}.json')

    import json
    metadata_filename = f'grasp_data/info/{cat}/{cat}{idx:03}.json'
    metadata = json.load(open(metadata_filename,'r'))
    scale = metadata['scale']
    obj = Object(path_to_obj_mesh, cat, scale=scale)
    scene.add_geometry(obj.mesh, geom_name='object')

def load_object_pointcloud(idx):
    # prepare pointclouds
    # pcl_filename = f"grasp_data/pcls/{cat}/{cat}{idx:03}.pkl"
    pcl_filename = f"../grasp_network_corrector/data/pcs/{cat}/{cat}{idx:03}.pkl"
    pcs_numpy = pickle.load(open(pcl_filename,'rb'))
    pcs = pcs_numpy['pcs']
    pc_index = np.random.randint(low=0, high=999, size=1)[0]
    pc = pcs[pc_index]
    pc = pcs[0:10]
    pc = np.concatenate(pc)
    scene.add_geometry(trimesh.points.PointCloud(vertices=pc))




# scene = Scene()
# scene.add_geometry(trimesh.points.PointCloud(vertices=pcs_numpy))
# scene.add_geometry(obj.mesh, geom_name='object')

for i in range(21):
    # print(i)
    scene = Scene()
    # load_object_mesh(i)
    load_object_pointcloud(i)

    mask = meta==i
    # print(mask)
    # exit()
    qs = quaternions[mask]
    ts = translations[mask]
    ls = labels[mask]
    
    for grasp_idx in range(min(50,len(qs))):
        quaternion = qs[grasp_idx]
        translation = ts[grasp_idx]
        gripper = utils.gripper_bd(quality=ls[grasp_idx])
        r = R.from_quat(quaternion)
        transform = np.eye(4)
        transform[:3,:3] = r.as_matrix()
        transform[:3,3] = translation
        gripper.apply_transform(transform)
        scene.add_geometry(gripper, geom_name='gripper')

    scene.show()



