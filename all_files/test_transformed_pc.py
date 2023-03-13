import numpy as np
from pathlib import Path

#from trimesh.permutate import transform
from graspsampler.common import PandaGripper, Scene, Object
import matplotlib.pyplot as plt
import trimesh.transformations as tra
import trimesh
import pickle
from graspsampler.utils import gripper_bd

from graspsampler.GraspSampler import GraspSampler


data = pickle.load(open('sample_check_data.pkl', 'rb'))

obj_filename = f'assets/sample_files/box001.stl'
graspsampler = GraspSampler(seed=10)

for k in range(20):
    graspsampler.update_object(obj_filename=obj_filename, name=obj_filename)
    gripper = gripper_bd()
    graspsampler.scene.add_geometry(gripper, geom_name='gripper_bd', transform=data['transforms'][k])

    for i, p in enumerate(data['gripper_pc'][k]):
        point_sphere = trimesh.primitives.Sphere(center=p, radius=0.001)
        point_sphere.visual.face_colors = [255,0,0,255]
        graspsampler.scene.add_geometry(point_sphere, f'point_{i}')

    pc = trimesh.points.PointCloud(data['pc'][k].T)
    graspsampler.scene.add_geometry(pc, 'pc')


    graspsampler.grasp_visualize(transform=data['transforms'][k],
                                coordinate_frame=True,
                                grasp_debug_items=True,
                                other_debug_items=True,
                                point=None,
                                origin = data['translations'][k])

    # graspsampler.scene.show()

    graspsampler.scene.delete_geometry('gripper_bd')

    for i, p in enumerate(data['gripper_pc'][k]):
        graspsampler.scene.delete_geometry(f'point_{i}')

    graspsampler.scene.delete_geometry('pc')