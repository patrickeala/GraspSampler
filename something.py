#!/usr/bin/python3.8

# Generating Grasps and Point Clouds for real-world testing

# PATRICK! READ THIS SCRIPT CAREFULLY!

import numpy as np
from graspsampler.common import PandaGripper, Object
# import matplotlib.pyplot as plt
# import trimesh.transformations as tra
import trimesh
import json
import pickle
from graspsampler.GraspSampler import GraspSampler
import os

# cwd = os.getcwd()
# dir = f"{cwd}/grasp_data/mug/"
# cat = '005_tomato_soup_can'
# obj_filename = f"{dir}/{cat}/google_16k/textured.obj"
# # init graspsampler constructor
# graspsampler = GraspSampler(seed=5)
# # load object
# graspsampler.update_object(obj_filename=f'grasp_data/meshes/mug/mug008.obj', name='box', obj_scale=[0.16,0.2,0.16])

# print()
# face_count = len(graspsampler.obj.mesh.faces)
# colors = np.ones([face_count,4])*125
# graspsampler.obj.mesh.visual.face_colors = [255, 0, 0, 155]
# # graspsampler.scene.show()
# number_of_grasps = 1
# points, normals, transforms, origins, quaternions, alpha, standoffs, collisions, qualities_1, qualities_2 = graspsampler.sample_grasps(number_of_grasps=number_of_grasps, silent=False)
# graspsampler.grasp_visualize(transform=transforms[0],
#                         coordinate_frame=False,
#                         grasp_debug_items=True,
#                         other_debug_items=True,
#                         point=points[0])


def make_colors(colors, ind):
    colors[ind] = np.array([255,0,0,255])
    return colors

obj = trimesh.load( file_obj=f'grasp_data/meshes/mug/mug008.obj', force='mesh')
face_count = len(obj.faces)
vertices_count = len(obj.vertices)

colors= np.ones([face_count, 4])*255

colors2= np.ones([vertices_count, 4])*255

for i in range(100,110):
    colors = make_colors(colors, i)

# print(obj.vertices)

visual = trimesh.visual.ColorVisuals(mesh=obj, face_colors=colors)
obj.visual = visual
obj.apply_scale([0.16,0.2,0.16])

scene = trimesh.Scene()
scene.add_geometry(obj)

scene.show()