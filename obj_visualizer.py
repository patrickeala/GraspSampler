
# from experiment_utils import utils

# import torch
from fileinput import filename
from re import T
from tkinter import Scale
from turtle import distance
import numpy as np
import argparse
from pathlib import Path
import pickle
from graspsampler.GraspSampler import GraspSampler
from graspsampler.common import PandaGripper, Scene, Object
import json
from scipy.spatial.transform import Rotation as R
from graspsampler import utils
from graspsampler.common import PandaGripper, Scene
import trimesh
import io
from PIL import Image

# object = "sugar_box"

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

task_info = {
    "hammer":["hammer","handover","mash","tenderize","crush"],
    "fork":["dig","handover","poke","scoop","mix"],
    "scissor":["poke","handover","cut","slice","open"],
    "mug":["drink","handover","lift","shake","hammer"],
    "bottle":["drink","pour","handover","ladle","screw"],
    "bowl":["handover","drink","pour","mix","scoop"], # Not applicable
    "spatula":["poke","handover","lift","mix","flip"],
    "pan":["pour","scoop","lift","saute","handover"],  # "flatten"
}


def get_grasp_point(trans, standoff = 0.2):
    standoff_mat = np.eye(4)
    standoff_mat[2] = standoff
    new = np.matmul(trans,standoff_mat)
    return new[:3,3]

def prepare_questions(obj_class):
    tasks = task_info[obj_class]
    # TODO finish it this when it is finalized

def save_grasp_images(grasp_index, scene, obj_class, obj_index, distance=0.65):

    save_dir = f"semantic_dataset/{obj_class}/{obj_class}{obj_index:03}_grasp_images/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    transform = np.eye(4)
    transform[:3,:3] = R.from_euler('Y', 0, degrees=True).as_matrix()
    point = [0,0,0]
    new_transform = scene.camera.look_at(points=[point],distance=distance,rotation=transform)
    scene.camera_transform = new_transform
    png = scene.save_image(resolution=[640*2, 480*2],
                        visible=True)
    with open(f"{save_dir}/grasp{grasp_index:05}_view1", 'wb') as f:
        f.write(png)
        f.close()

    transform = np.eye(4)
    transform[:3,:3] = R.from_euler('Y', 120, degrees=True).as_matrix()
    point = [0,0,0]
    new_transform = scene.camera.look_at(points=[point],distance=distance,rotation=transform)
    scene.camera_transform = new_transform
    png = scene.save_image(resolution=[640*2, 480*2],
                        visible=True)
    with open(f"{save_dir}/grasp{grasp_index:05}_view2", 'wb') as f:
        f.write(png)
        f.close()
    
    transform = np.eye(4)
    transform[:3,:3] = R.from_euler('X', -60, degrees=True).as_matrix()
    point = [0,0,0]
    new_transform = scene.camera.look_at(points=[point],distance=distance,rotation=transform)
    scene.camera_transform = new_transform
    png = scene.save_image(resolution=[640*2, 480*2],
                    visible=True)
    with open(f"{save_dir}/grasp{grasp_index:05}_view3", 'wb') as f:
        f.write(png)
        f.close()



# obj_classes  = ["pan","spatula","bowl","bottle","mug","fork","hammer","scissor"]
obj_classes = ["box", "cylinder"]
for obj_class in obj_classes:
    num_obj = 20
    if obj_class == "mug":
        num_obj = 21
    for idx in range(num_obj):
        obj_name = f"{obj_class}{idx:03}"
        obj_filename = f"grasp_data/meshes/{obj_class}/{obj_name}.obj"
        if obj_class in ["box", "cylinder"]:
            obj_filename = f"grasp_data/meshes/{obj_class}/{obj_name}.stl"
        
        metadata_filename = f'grasp_data/info/{obj_class}/{obj_name}.json'
        metadata = json.load(open(metadata_filename,'r'))


        # load object
        obj_model = Object(filename = obj_filename, scale = metadata["scale"], name = "grasp_obj")
        scene = Scene()
        scene.add_mesh(obj_model.mesh)
        scene.show()
        # scene.add_geometry(obj_mesh)
     

# label_obj(1)
# from joblib import Parallel, delayed
# Parallel(n_jobs=3)(delayed(label_obj)(i) for i in range(8))


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
# plt.show()<img src=”(your image URL here)”>

