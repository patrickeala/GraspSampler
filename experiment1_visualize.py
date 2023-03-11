
# from experiment_utils import utils

import torch
import numpy as np
import argparse
# from models.models import GraspSamplerDecoder, GraspEvaluator
# from models.quaternion import quaternion_mult
# from utils import density_utils
import pickle
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path
import time
from scipy.spatial.transform import Rotation as R
from graspsampler import utils
from graspsampler.common import PandaGripper, Scene


# [504 747 237 202 548 248  31 758 752 802 407 875 725 285 341 831 801 938 265 
# 406 882 993 931 239 588 818 523 673 471 631 869 812 724 309 554 150525 218 
# 23 726 193 570 979 362 549 872 748 947 773 255]

info, args = pickle.load(open('results/1641738054/info', 'rb'))

print(info.keys())
# seq, B, ...
N = 50
success_last = info['success'][-1, :]
# print(success_last[np.argsort(success_last)[::-1][:N]])
# mask = np.argsort(success_last)[::-1][:N]
# print(mask)
# mask = [504 747 548 248 758 (285) 831 (265) 588 818 (523) 471 525 193 979 773 255]
mask = [554]   
translations_init = info['translations'][0, mask, :]
quaternions_init = info['quaternions'][0, mask, :]
translations_final = info['translations'][-1, mask, :]
quaternions_final = info['quaternions'][-1, mask, :]
print(success_last[mask])
print(translations_final)
print(quaternions_final)

# prepare pointclouds
pcs_numpy, obj_pose_relative = pickle.load(open(f'experiment1_pcs/004_sugar_box.pkl','rb'))
pcs_numpy = np.concatenate([pcs_numpy[0],pcs_numpy[1]])

import trimesh
scene = Scene()
scene.add_geometry(trimesh.points.PointCloud(vertices=pcs_numpy))

for i in range(translations_init.shape[0]):
    gripper = utils.gripper_bd(0)
    r = R.from_quat(quaternions_init[i])
    transform = np.eye(4)
    transform[:3,:3] = r.as_matrix()
    transform[:3,3] = translations_init[i]
    gripper.apply_transform(transform)
    scene.add_geometry(gripper)

number_of_grasps = 50
# panda = PandaGripper()
# panda.apply_transformation(transform)
# scene.add_gripper(panda)
for i in range(translations_final.shape[0]):
    
    gripper = utils.gripper_bd(1)
    r = R.from_quat(quaternions_final[i])
    transform = np.eye(4)
    transform[:3,:3] = r.as_matrix()
    transform[:3,3] = translations_final[i]

    gripper.apply_transform(transform)
    scene.add_geometry(gripper)

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


scene.show()

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