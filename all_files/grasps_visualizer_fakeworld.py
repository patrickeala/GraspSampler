
# from experiment_utils import utils

# import torch
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

dataset = 'fakeworld_grasp_data_generated'
# dataset = 'fakeworld_vel_grasp_data_generated'

data_path = f"/home/user/isaac/{dataset}/{cat}/{cat}{idx:03}_isaac_sim2fake/all_info_{trial}.pkl"
data = pickle.load(open(data_path,'rb'))
# print(len(data))

quaternions, translations, isaac_sim_labels, isaac_fake_labels, obj_pose_relative = data
print(f"Total grasps: {quaternions.shape[0]}")
sim_mask = np.array(isaac_sim_labels, dtype=bool)
quaternions = quaternions[sim_mask]
translations = translations[sim_mask]
# print(quaternions.shape[0])
# print(translations.shape)
# print(np.unique(isaac_sim_labels, return_counts=True))
# print(isaac_fake_labels.shape)
# print(obj_pose_relative)
print(f"Sim+ grasps: {np.sum(isaac_sim_labels)}")
print(f"Fake+ grasps: {np.sum(isaac_fake_labels)}")
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
scale = utils.get_scale(f'{path_to_assets}/info/{cat}/{obj_name}.json')
    

obj = Object(path_to_obj_mesh, cat, scale=scale)


import trimesh
scene = Scene()
# scene.add_geometry(trimesh.points.PointCloud(vertices=pcs_numpy))
scene.add_geometry(obj.mesh, geom_name='object')


for grasp_idx in range(min(100,len(quaternions))):
# for grasp_idx in range(100):
# for grasp_idx in [5,8]: # mug 1
    # print(f"{obj_name} {grasp_idx}")
# for grasp_idx in range(len(quaternions)):
    # main_grasps_data = np.load(f'{data_path}/main_grasps.npz')
    
    # main_grasps_data = np.load(f'{data_path}/main_grasps.npz')
    # main_grasp_quaternion = main_grasps_data["quaternions"][grasp_idx]
    # main_grasp_translation = main_grasps_data["translations"][grasp_idx]

    # sub_grasps_data = np.load(f'{data_path}/{grasp_idx:08}.npz')
    # sub_grasps_quaternions = sub_grasps_data["quaternions"]
    # sub_grasps_translations = sub_grasps_data["translations"]
    # sub_grasps_labels = sub_grasps_data["isaac_labels"]

    quaternion = quaternions[grasp_idx]
    translation = translations[grasp_idx]
    # print(quaternion)
    # print(translation)
    # continue

    # add main grasp
    # if isaac_fake_labels[grasp_idx] == 0:
        # continue
    gripper = utils.gripper_bd(quality=isaac_fake_labels[grasp_idx], opacity=0.75)
    r = R.from_quat(quaternion)
    transform = np.eye(4)
    transform[:3,:3] = r.as_matrix()
    transform[:3,3] = translation
    gripper.apply_transform(transform)
    scene.add_geometry(gripper, geom_name='gripper')
    # scene.show()
    # scene.delete_geometry('gripper')

    # for i in range(100):
    #     # if sub_grasps_labels[i] == 1:
    #     #     continue
    #     gripper = utils.gripper_bd(sub_grasps_labels[i])
    #     r = R.from_quat(sub_grasps_quaternions[i])
    #     transform = np.eye(4)
    #     transform[:3,:3] = r.as_matrix()
    #     transform[:3,3] = sub_grasps_translations[i]
    #     gripper.apply_transform(transform)
    #     scene.add_geometry(gripper)
        
scene.show()

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