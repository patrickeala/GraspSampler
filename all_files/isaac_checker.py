import numpy as np
from pathlib import Path

#from trimesh.permutate import transform
from graspsampler.common import PandaGripper, Scene, Object
import matplotlib.pyplot as plt
import trimesh.transformations as tra
import trimesh
import pickle
from graspsampler.utils import gripper_bd
from graspsampler.utils import trans_matrix
from graspsampler.GraspSampler import GraspSampler


objs = {
    0: '004_sugar_box',
    1: '005_tomato_soup_can',
    2: '006_mustard_bottle',
    3: '007_tuna_fish_can',
    4: '010_potted_meat_can',
    5: '014_lemon',
    6: '025_mug',
    7: '026_sponge',
    8: '061_foam_brick',
    9: '065_j_cups'
}
scene = Scene()
gripper = PandaGripper()
i = 0

obj_filename = f'assets/meshes_10_objects/{objs[i]}/google_16k/textured.obj'
gripper_small = gripper_bd()
data = pickle.load( open(f'isaac_test_10_meshes/{objs[i]}_top1.pkl', 'rb') )
obj = Object(obj_filename, shift_to_center=False, name='obj')
obj_pose_relative = obj.get_obj_mesh_mean()
print(obj_pose_relative)

obj.apply_transform(trans_matrix(euler=[0,0,0], translation=-obj_pose_relative))
# scene.add_gripper(gripper=gripper)
scene.add_object(obj, face_colors=[0, 255, 0, 170])

for k in range(10):

    scene.add_geometry(gripper_small, geom_name='gripper_bd', transform=data['transforms'][k])


scene.add_coordinate_frame()
scene.show()
