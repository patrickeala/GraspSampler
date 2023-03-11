
# from experiment_utils import utils

import torch
from gettext import translation
import numpy as np
import argparse
# from models.models import GraspSamplerDecoder, GraspEvaluator
# from models.quaternion import quaternion_mult
# from utils import density_utils
import pickle
from graspsampler.GraspSampler import GraspSampler
from graspsampler.common import PandaGripper, Scene, Object
# from torch.utils.data import DataLoader, TensorDataset
# import torch.nn.functional as F
# from tqdm.auto import tqdm
# from pathlib import Path
# import time
from scipy.spatial.transform import Rotation as R
from graspsampler import utils
from graspsampler.common import PandaGripper, Scene


# cat = "bowl"
import sys
cat = sys.argv[-2]
idx = int(sys.argv[-1])
# print(cat)
# print(idx)
# idx = 2
# grasp_idx = 0
trial = 1
dataset = 'fakeworld_grasp_data_generated_hardcoded'

# dataset = 'fakeworld_grasp_data_table'
# dataset = 'fakeworld_grasp_data_friction'
# dataset = 'fakeworld_grasp_data_generated'
# dataset = 'fakeworld_vel_grasp_data_generated'

data_path = f"/home/gpupc2/GRASP/isaac/{dataset}/{cat}/{cat}{idx:03}_isaac_sim2fake/all_info_{trial}.pkl"
data = pickle.load(open(data_path,'rb'))
# print(len(data))
print(data_path)
quaternions, translations, isaac_sim_labels, isaac_fake_labels, obj_pose_relative = data
print(f"Total grasps: {quaternions.shape[0]}")


# sim_mask = np.array(isaac_sim_labels, dtype=bool)
# fake_mask = np.array(isaac_fake_labels, dtype=bool)
# all_mask = np.logical_or(sim_mask,fake_mask)


# quaternions = quaternions[all_mask]
# translations = translations[all_mask]
# isaac_fake_labels = isaac_fake_labels[all_mask]
# print(f"Sim+ grasps: {np.sum(isaac_sim_labels)}")
# print(f"Fake+ grasps: {np.sum(isaac_fake_labels)}")
# print(f"Both Sim+ and Fake+ grasps: {np.sum(all_mask)}")
# exit()


# prepare pointclouds
# pcl_filename = f"grasp_data/pcls/{cat}/{cat}{idx:03}.pkl"
# pcs_numpy = pickle.load(open(pcl_filename,'rb'))
# pcs_numpy = np.concatenate([pcs_numpy[0],pcs_numpy[1],pcs_numpy[2],pcs_numpy[3]])


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



# import open3d as o3d
# filepath= "/home/gpupc2/GRASP/dgcnn.pytorch/outputs/patrick_test/visualization/knife/knife_0_pred_.ply"
# pcd_read = o3d.io.read_point_cloud(filepath)
# print(np.asarray(pcd_read.points))
# exit()


seg_file = np.load('/home/gpupc2/GRASP/grasper/pat_semantic/hammer.npz')
pc = seg_file['data'].squeeze().transpose()
pc_torch = torch.from_numpy(pc)


pred = seg_file['pred'].squeeze()


pred = torch.from_numpy(pred)
pred -= torch.min(pred)

colors = torch.ones((pc.shape[0], 4))
colors[:,1] = pred
colors[:,0] -= pred
colors[:,2] -= pred

cent1 = torch.mean(pc_torch[torch.where(pred==0)],axis=0)
cent2 = torch.mean(pc_torch[torch.where(pred==1)],axis=0)


# cent1 = torch.from_numpy(cent1)
# cent2 = torch.from_numpy(cent2)
# pred_torch = torch.from_numpy(pred)


def get_transformed_handle(transform):
    gripper_pts = np.array([
        [-0.0, 0.0000599553131906, 0.0672731392979622],
        [-0.0, 0.0000599553131906, -0.0032731392979622]
    ])
    

    final_pts = gripper_pts @ np.linalg.inv(transform[:3,:3])
    final_pts += transform[:3,3]
    return torch.from_numpy(final_pts)

def get_semantic_score(transform, pc, label, cent1, cent2):

    handle_pts = get_transformed_handle(transform.numpy())

    
    



    handle = utils.gripper_handle(1)
    handle.apply_transform(transform)
    scene.add_geometry(handle, geom_name='handle')





    handle_orig = trimesh.path.Path3D(entities=[trimesh.path.entities.Line([0, 1])],
                                            vertices = handle_pts)
    scene.add_geometry(handle_orig, geom_name='handle')


    # return score



import trimesh
scene = Scene()
scene.add_geometry(trimesh.points.PointCloud(vertices=pc, colors=colors))
# scene.show()
# exit()
# scene.add_geometry(obj.mesh, geom_name='object')



for grasp_idx in range(min(100,len(quaternions))):


    quaternion = quaternions[grasp_idx]
    translation = translations[grasp_idx]


    # gripper = utils.gripper_bd(quality=isaac_fake_labels[grasp_idx])
    r = R.from_quat(quaternion)
    transform = np.eye(4)
    transform[:3,:3] = r.as_matrix()
    transform[:3,3] = translation


    trans = torch.from_numpy(transform)
    score = get_semantic_score(trans, pc_torch, pred, cent1, cent2)
    # get projected point
    # score = get_semantic_score(transform, centroids)





    # gripper = utils.gripper_bd()
    # gripper.apply_transform(transform)
    # scene.add_geometry(gripper, geom_name='gri
    # 
    # pper')
    scene.show()

    # pred_torch = torch.from_numpy(pred)
    exit()
# scene.show()

# exit()
# scene.add_geometry(obj_mesh)





# number_of_grasps = 50
# panda = PandaGripper()
# panda.apply_transformation(transform)
# scene.add_gripper(panda)


# for i in range(translations_final.shape[0]):
    
#     gripper = utils.gripper_bd(1)
#     r = R.from_quat(quaternions_final[i])
#     transform = np.eye(4)
#     transform[:3,:3] = r.as_matrix()
#     transform[:3,3] = translations_final[i]

#     gripper.apply_transform(transform)
#     scene.add_geometry(gripper)

    # transforms= utils.perturb_transform(transform, number_of_grasps, 
    #                 min_translation=(-0.01,-0.01,-0.01),
    #                 max_translation=(0.01,0.01,0.01),
    #                 min_rotation=(-0.125,-0.125,-0.125),
    #                 max_rotation=(+0.125,+0.125,+0.125))
    # print(len(transforms))

    # for _trans in transforms:
    #     gripper = utils.gripper_bd(0)
    #     gripper.apply_transform(_trans)
    #     scene.add_geometry(gripper)


# for i in range(all_trans.shape[1]):
#     gripper = utils.gripper_bd(1)
#     r = R.from_euler('XYZ', all_eulers[-1][i])
#     transform = np.eye(4)
#     transform[:3,:3] = r.as_matrix()
#     transform[:3,3] = all_trans[-1][i]
#     gripper.apply_transform(transform)
#     scene.add_geometry(gripper)

# pc = trimesh.points.PointCloud(vertices=pcs_numpy)
# scene.add_geometry(pc)
# scene.show()


# import matplotlib.pyplot as plt
# all_trans_v = np.mean(all_trans_v, axis=1)
# print(all_trans_v.shape)
# plt.plot(all_trans_v[:,0], label='x')
# plt.plot(all_trans_v[:,1], label='y')
# plt.plot(all_trans_v[:,2], label='z')
# # plt.show()

# # all_quat_v = all_quat_v.squeeze(1)
# # plt.plot(all_quat_v[:,0], 'r--')
# # plt.plot(all_quat_v[:,1], 'r--')
# # plt.plot(all_quat_v[:,2], 'r--')
# # plt.plot(all_quat_v[:,3], 'r--')
# # plt.show()

# all_euler_v = np.mean(all_euler_v, axis=1)
# plt.plot(all_euler_v[:,0], 'r--')
# plt.plot(all_euler_v[:,1], 'r--')
# plt.plot(all_euler_v[:,2], 'r--')
# plt.legend()
# plt.show()



# all_success = all_success.squeeze()
# print(all_success.shape)
# plt.plot(all_success)
# plt.show()