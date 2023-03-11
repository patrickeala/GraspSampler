
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
import trimesh
import json

# object = "sugar_box"
def tc_rotation_matrix(az, el, th, batched=False):
    if batched:

        cx = torch.cos(torch.reshape(az, [-1, 1]))
        cy = torch.cos(torch.reshape(el, [-1, 1]))
        cz = torch.cos(torch.reshape(th, [-1, 1]))
        sx = torch.sin(torch.reshape(az, [-1, 1]))
        sy = torch.sin(torch.reshape(el, [-1, 1]))
        sz = torch.sin(torch.reshape(th, [-1, 1]))

        ones = torch.ones_like(cx)
        zeros = torch.zeros_like(cx)

        rx = torch.cat([ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx],
                       dim=-1)
        ry = torch.cat([cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy],
                       dim=-1)
        rz = torch.cat([cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones],
                       dim=-1)

        rx = torch.reshape(rx, [-1, 3, 3])
        ry = torch.reshape(ry, [-1, 3, 3])
        rz = torch.reshape(rz, [-1, 3, 3])

        return torch.matmul(rz, torch.matmul(ry, rx))
    else:
        cx = torch.cos(az)
        cy = torch.cos(el)
        cz = torch.cos(th)
        sx = torch.sin(az)
        sy = torch.sin(el)
        sz = torch.sin(th)

        rx = torch.stack([[1., 0., 0.], [0, cx, -sx], [0, sx, cx]], dim=0)
        ry = torch.stack([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dim=0)
        rz = torch.stack([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dim=0)

        return torch.matmul(rz, torch.matmul(ry, rx))
# object_to_code = {
#     "sugar_box"      :"004_sugar_box",
#     "bowl"           :"024_bowl",
#     "tomato_soup_can":'005_tomato_soup_can', 
#     "potted_meat_can":'010_potted_meat_can', 
#     "mug"            :'025_mug', 
#     "foam_brick"     : '061_foam_brick', 
#     "j_cups"         :'065_j_cups',
#     "sponge"         :'026_sponge'
#     }

# data_path = f'/home/user/isaac/experiments7/{object_to_code[object]}'

# graspnet_initial_grasps_fname = f'{data_path}/grasps_graspnet_initial.npz'
# graspnet_final_grasps_fname = f'{data_path}/grasps_graspnet_final.npz'

# data = np.load(graspnet_initial_grasps_fname)

# translations_init = data["translations"]
# quaternions_init = data["quaternions"]
# # distances = pickle.load(open(f"{data_path}/grasps_graspnet_initial_distances.pkl","rb"))

# # prepare pointclouds
# obj_filename = f"assets/meshes_10_objects/{object_to_code[object]}/google_16k/textured.obj"


# # prepare pointclouds
# pcs_numpy = pickle.load(open(f'{data_path}/pcs.pkl','rb'))
# pcs_numpy = np.concatenate([pcs_numpy[0],pcs_numpy[1],pcs_numpy[2],pcs_numpy[3]])

def get_grasp_point(trans, standoff = 0.2):
    standoff_mat = np.eye(4)
    standoff_mat[2] = standoff
    new = np.matmul(trans,standoff_mat)
    return new[:3,3]

obj_class = "hammer"
idx = 0
obj_name = f"{obj_class}{idx:03}"
grasp_filename = f"../isaac/grasps_sampled_for_positive_grasps/{obj_class}/{obj_name}_isaac/positive_grasps.npz"
grasp_data = np.load(grasp_filename)
translations_init = grasp_data["translations"]
quaternions_init = grasp_data["quaternions"]
obj_pose_relative = grasp_data["obj_pose_relative"]
# translations_init = translations_init[:,] + obj_pose_relative
pc_filename = f"grasp_data/pcls/{obj_class}/{obj_name}.pkl"
pc_real = np.array(pickle.load(open(pc_filename, 'rb')))
gripper_control_points = np.array([[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00, 1],
                                    [ 0.0000000e+00,  0.0000000e+00 , 7.5273141e-02, 1],
                                    [-5.2687433e-02,  5.9955313e-05 , 7.5273141e-02, 1],
                                    [-5.2687433e-02 , 5.9955313e-05, 1.0527314e-01, 1],
                                    [-5.2687433e-02,  5.9955313e-05 , 7.5273141e-02, 1],
                                    [ 5.2687433e-02, -5.9955313e-05 , 7.5273141e-02, 1],
                                    [ 5.2687433e-02, -5.9955313e-05 , 1.0527314e-01, 1],
                                    ])
import torch
# gripper_pc = torch.Tensor(gripper_control_points.copy()[:,:3])
# gripper_pc = torch.matmul(gripper_pc[:,:3], rot_mat.permute(0,2,1))
# #print(gripper_pc.shape, translations.unsqueeze(1).expand(-1, gripper_pc.shape[1],-1).shape)
# gripper_pc += translations.unsqueeze(1).expand(-1, gripper_pc.shape[1],-1)

pc_real = np.concatenate([pc_real])

obj_filename = f"grasp_data/meshes/{obj_class}/{obj_class}{idx:03}.obj"
metadata_filename = f'grasp_data/info/{obj_class}/{obj_class}{idx:03}.json'
metadata = json.load(open(metadata_filename,'r'))
sampler = GraspSampler()
# load object
sampler.update_object(obj_filename=obj_filename, name=obj_filename, obj_scale=metadata['scale'])
print(sampler.obj.obj_mesh_mean)


scene = Scene()
scene.add_geometry(sampler.obj.mesh)
scene.add_geometry(trimesh.points.PointCloud(vertices=pc_real))
# index = np.random.choice(translations_init.shape[0], 10000, replace=False)  
# translations_init = translations_init[index]

# quaternions_init = quaternions_init[index]

upper_threshold = 0
lower_threshold = 0
dimension = 1

scene = Scene()
scene.add_geometry(sampler.obj.mesh)
scene.add_geometry(trimesh.points.PointCloud(vertices=pc_real))
import copy
_gripper = copy.deepcopy(sampler.gripper)

# scene.add_geometry(obj_mesh)
for i in range(98,99):

    r = R.from_quat(quaternions_init[i])
    transform = np.eye(4)
    transform[:3,:3] = r.as_matrix()
    transform[:3,3] = translations_init[i]
    # grasp_point = get_grasp_point(transform,standoff=0.1)
    # # print(translations_init[i], grasp_point)
    # if grasp_point[dimension] > lower_threshold :
    #     continue
    gripper_pc = torch.Tensor(gripper_control_points.copy()[:,:3])
    grasp = np.array([transform])
    translations = torch.Tensor(grasp[:, :3, 3])

    eulers = R.from_matrix(grasp[:, :3,:3]).as_euler(seq='xyz')
    eulers = torch.Tensor(eulers.copy())
    rot_mat = tc_rotation_matrix(eulers[:,0], eulers[:,1], eulers[:,2], True)

    gripper_pc = torch.matmul(gripper_pc[:,:3],rot_mat.permute(0,2,1))
#print(gripper_pc.shape, translations.unsqueeze(1).expand(-1, gripper_pc.shape[1],-1).shape)
    gripper_pc += translations.unsqueeze(1).expand(-1, gripper_pc.shape[1],-1)
    scene.add_geometry(trimesh.points.PointCloud(vertices=gripper_pc[0],colors=[0,255,0,255]))

    gripper = utils.gripper_bd(0)
    gripper.apply_transform(transform)
    scene.add_geometry(gripper)
    _gripper.apply_transformation(transform=transform)
    scene.add_gripper(_gripper)
scene.show()

# scene = Scene()
# scene.add_geometry(trimesh.points.PointCloud(vertices=pc_real))


# for i in range(1000):

#     r = R.from_quat(quaternions_init[i])
#     transform = np.eye(4)
#     transform[:3,:3] = r.as_matrix()
#     transform[:3,3] = translations_init[i]
#     grasp_point = get_grasp_point(transform,standoff=0.1)
#     # print(translations_init[i], grasp_point)
#     if grasp_point[dimension] <=  upper_threshold :
#         continue
#     gripper = utils.gripper_bd(0)
#     gripper.apply_transform(transform)
#     scene.add_geometry(gripper)
# scene.show()

    # gripper = utils.gripper_bd(0)
    # gripper.apply_transform(grasp_point)
    # scene.add_geometry(gripper)
    # break
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