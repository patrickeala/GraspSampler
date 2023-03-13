
# from experiment_utils import utils

# import torch
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

cat = "mug"
idx = 14
# grasp_idx = 0
data_path = f'../isaac/experiment4/{cat}/{cat}{idx:03}/'
heuristic_initial_grasps_fname = f'{data_path}/grasps_heuristics_final.npz'
# data_path = "/home/user/isaac/grasps_sampled_for_positive_grasps/mug/mug000_isaac"

# prepare pointclouds
pcl_filename = f"grasp_data/pcls/{cat}/{cat}{idx:03}.pkl"
pcs_numpy = pickle.load(open(pcl_filename,'rb'))
pcs_numpy = np.concatenate([pcs_numpy[0],pcs_numpy[1],pcs_numpy[2],pcs_numpy[3]])


import trimesh
scene = Scene()
scene.add_geometry(trimesh.points.PointCloud(vertices=pcs_numpy))

# for grasp_idx in range(1):

main_grasps_data = np.load(heuristic_initial_grasps_fname)
main_grasp_quaternion = main_grasps_data["quaternions"]
main_grasp_translation = main_grasps_data["translations"]

    # sub_grasps_data = np.load(heuristic_initial_grasps_fname)
    # sub_grasps_quaternions = sub_grasps_data["quaternions"]
    # sub_grasps_translations = sub_grasps_data["translations"]
    # sub_grasps_labels = sub_grasps_data["isaac_labels"]

    # add main grasp
idx = 19
gripper = utils.gripper_bd()
r = R.from_quat(main_grasp_quaternion[idx])
transform = np.eye(4)
transform[:3,:3] = r.as_matrix()
transform[:3,3] = main_grasp_translation[idx]
gripper.apply_transform(transform)
scene.add_geometry(gripper)
scene.show()
exit()
for i in range(0,len(main_grasp_quaternion),20):
    for j in range(i+15,i+20):
        gripper = utils.gripper_bd()
        r = R.from_quat(main_grasp_quaternion[j])
        transform = np.eye(4)
        transform[:3,:3] = r.as_matrix()
        transform[:3,3] = main_grasp_translation[j]
        gripper.apply_transform(transform)
        scene.add_geometry(gripper)
    scene.show()
    scene = Scene()
    scene.add_geometry(trimesh.points.PointCloud(vertices=pcs_numpy))

    # for i in range(100):
#         # if sub_grasps_labels[i] == 1:
#         #     continue
#         gripper = utils.gripper_bd(1)
#         r = R.from_quat(sub_grasps_quaternions[i])
#         transform = np.eye(4)
#         transform[:3,:3] = r.as_matrix()
#         transform[:3,3] = sub_grasps_translations[i]
#         gripper.apply_transform(transform)
#         scene.add_geometry(gripper)
        
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