#!/usr/bin/python3.8

# Author: Tasbolat Taunyazov
# TUTORIAL 3: Manipulating point cloud manager

import numpy as np
from graspsampler.common import PandaGripper, Scene, Object
from graspsampler.PointCloudManager import PointCloudManager
import matplotlib.pyplot as plt
import trimesh.transformations as tra

# prepare object
box = Object(filename='assets/sample_files/box000.stl', name='box')

# define pc_manager
# NOTE: pointcloud manager operates on PyRender scene (Not Trimesh scene)
pc_manager = PointCloudManager(obj_mesh=box.mesh, fit_coefficient=5)

# when object is added, it is placed away to fit to the scene with fit_coeff
pc_manager.update_object(obj_mesh=box.mesh)
pc_manager.set_obj_pose(np.eye(4), adjust_object_to_fit=True)
pc_manager.view_scene()

import time

color, depth, pc, _ = pc_manager.render_pc()

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(color)
ax[1].imshow(depth)
plt.show()

print('pc shape')
print(pc.shape)
print('pc stat:')
# point cloud is not normalized
print(pc.min(), pc.max(), pc.mean())

pc_manager.view_pointcloud(pc)

# rotate object
pc_manager.update_object(obj_mesh=box.mesh)
pc_manager.set_obj_pose(tra.euler_matrix(0, np.pi/6, np.pi/2))

color, depth, pc, _ = pc_manager.render_pc(full_pc=True)
print(pc.min(), pc.max(), pc.mean())
pc_manager.view_pointcloud(pc)

# import trimesh
# import numpy as np

# pc = np.random.random([8,3])
# print(pc.shape)
# print(pc)

# pc_mesh = trimesh.points.PointCloud(pc)
# pc_mesh.show()


# import numpy as np
# import trimesh
# import pyrender
# from graspsampler.common import PandaGripper, Scene, Object
# from graspsampler.PointCloudManager import PointCloudManager


# # box = trimesh.primitives.Box()
# # scene = trimesh.Scene()
# # scene.add_geometry(box)
# # scene.show()

# box = Object(filename='assets/sample_files/box000.stl', name='box')
# pc_manager = PointCloudManager(obj_mesh=box.mesh, fit_coefficient=20)
# pyr_scene = pyrender.Scene()
# pyr_mesh = pyrender.Mesh.from_trimesh(box.mesh)
# pyr_scene.add(pyr_mesh)
# pyrender.Viewer(pyr_scene)


# box = trimesh.points.PointCloud(np.random.random([1000,3]))
# scene = trimesh.Scene()
# scene.add_geometry(box)
# scene.show()