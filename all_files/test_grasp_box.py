import numpy as np
from graspsampler.common import PandaGripper, Scene, Object
import trimesh.transformations as tra
from graspsampler.GraspSampler import GraspSampler
import pickle as pickle
from scipy.spatial.transform import Rotation as R
import json

<<<<<<< HEAD
category = "fork"
idx = 8
obj_name = f"{category}{idx:03}"
obj_filename = f"grasp_data/meshes/{category}/{obj_name}.obj"
# grasp_filename = f"/home/user/isaacgym/python/IsaacGymGrasp/box_grasps/grasps/{obj_name}.pkl"
grasp_filename = f"../isaac/grasp_data_generated/{category}/{obj_name}_isaac/main1_rest.npz"
metadata_filename = f'grasp_data/info/{category}/{category}{idx:03}.json'
metadata = json.load(open(metadata_filename,'r'))
=======
category = "scissor"
idx = 0
obj_name = f"{category}{idx:03}"
obj_filename = f"grasp_data/meshes/{category}/{obj_name}.obj"
# grasp_filename = f"/home/user/isaacgym/python/IsaacGymGrasp/box_grasps/grasps/{obj_name}.pkl"
# grasp_filename = f"grasp_data_generated/{category}/{obj_name}/main1.npz"

for grasp_idx in range(1000):
    # grasp_filename = f"../isaac_clean/grasps_sampled_for_positive_grasps/{category}/{obj_name}_issac/{grasp_idx:08}.npz"
    grasp_filename =  f"/home/crslab/GRASP/isaac/grasps_sampled_for_positive_grasps/scissor/scissor000_isaac/{grasp_idx:08}.npz"
   # grasp_filename = f"../isaac_clean/grasps_sampled_for_positive_grasps/{category}/{obj_name}_issac/{grasp_idx:08}.npz"


>>>>>>> fdc2ad5bb69cd9ae724f86417d268c2d7e06f696
# indecies_filename = f"/home/user/isaacgym/python/IsaacGymGrasp/box_mismatch_indcs/{obj_name}.npy"
# with open(indecies_filename, "rb") as fh:
    # index_of_grasps = np.load(fh)
# with open(grasp_filename, "rb") as fd:
    data = np.load(grasp_filename)


# grasps_translations = data["translations"][index_of_grasps.flatten()]
# grasps_quaternions = data["quaternions"][index_of_grasps.flatten()]
<<<<<<< HEAD
grasps_translations = data["translations"]
grasps_quaternions = data["quaternions"]
is_promising = data["isaac_labels"]
=======
    grasps_translations = data["translations"]
    grasps_quaternions = data["quaternions"]
    # is_promising = data["is_promising"]
    isaac_labels = data["isaac_labels"]
>>>>>>> fdc2ad5bb69cd9ae724f86417d268c2d7e06f696
# qualities_1 = np.array(data['qualities_1'])
# qualities_2 = np.array(data['qualities_2'])
# collisions  =   data['collisions'] 

<<<<<<< HEAD
grasps = []
for grasp_translation, grasps_quaternion, promising in zip(grasps_translations, grasps_quaternions, is_promising):
    if promising:
        transform = np.eye(4)
        r = R.from_quat(grasps_quaternion)
        transform[:3,:3] = r.as_matrix()
        transform[:3,3] = grasp_translation
        grasps.append(transform)

index_to_check = 0#2507
# index_to_check = 26
# print("qualities_1: ",qualities_1[index_to_check])
# print("qualities_2: ",qualities_2[index_to_check])
# print("collision: ", collisions[index_to_check])
sampler = GraspSampler()
sampler.update_object(obj_filename=obj_filename, name='box',obj_scale=metadata["scale"])

print(grasps_quaternions[index_to_check])
print(grasps_translations[index_to_check])
for idx in range(len(grasps)):
    sampler.grasp_visualize(grasps[idx], coordinate_frame=True,grasp_debug_items=False,other_debug_items=False)
=======
    grasps = []
    for grasp_translation, grasps_quaternion, isaac_label in zip(grasps_translations, grasps_quaternions, isaac_labels):
        if isaac_label:
            transform = np.eye(4)
            r = R.from_quat(grasps_quaternion)
            transform[:3,:3] = r.as_matrix()
            transform[:3,3] = grasp_translation
            grasps.append(transform)
            

    # index_to_check = 0#2507
    # index_to_check = 26
    # print("qualities_1: ",qualities_1[index_to_check])
    # print("qualities_2: ",qualities_2[index_to_check])
    # print("collision: ", collisions[index_to_check])
    sampler = GraspSampler()
    sampler.update_object(obj_filename=obj_filename, name='box')
    for idx in range(len(grasps)):
        sampler.grasp_visualize(grasps[idx], coordinate_frame=True,grasp_debug_items=True,other_debug_items=False)
        break
>>>>>>> fdc2ad5bb69cd9ae724f86417d268c2d7e06f696
# sampler.grasp_visualize(data["transforms"][index_to_check], coordinate_frame=True,grasp_debug_items=True,other_debug_items=False)






