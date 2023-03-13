# from importlib.metadata import metadata
# from turtle import color
import numpy as np
import argparse
# from models.models import GraspSamplerDecoder, GraspEvaluator
# from models.quaternion import quaternion_mult
# from utils import density_utils
import pickle
# from graspsampler.GraspSampler import GraspSampler
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
# import open3d as o3d


obj_path = "/home/gpupc2/GRASP/grasper/assets/meshes_10_objects/004_sugar_box/google_16k/textured.obj"
obj_mesh = Object(filename=obj_path, name="sugar_box")
obj_mesh = obj_mesh.mesh

point_idx = np.random.choice(len(obj_mesh.vertices),3000,replace=False)
points = obj_mesh.vertices[point_idx]
points_color = obj_mesh.visual.to_color().vertex_colors[point_idx]
obj_pcl = trimesh.points.PointCloud(vertices=points, colors=points_color)
scene = Scene()
scene.add_geometry(obj_pcl)
# scene.show()

grasp_transforms = [
[[ 1,   0,   0.,   0., ],
 [ 0.  , 1. ,  0. ,  0. ],
 [ 0.  , 0. ,  1. , -0.2],
 [ 0.  , 0.  , 0.  , 1. ],],
 
 [[ 1,   0,   0.,   0., ],
 [ 0.  , 1. ,  0. ,  0.03 ],
 [ 0.  , 0. ,  1. , -0.17],
 [ 0.  , 0.  , 0.  , 1. ],],

 [[ 1.   ,       0.     ,     0.        , -0.01      ],
 [ 0.    ,      0.25881905,  0.96592583, -0.15      ],
 [ 0.    ,     -0.96592583,  0.25881905, -0.13     , ],
 [ 0.    ,      0.     ,     0.      ,    1.        ],],

 [[ 1.   ,       0.     ,     0.        , -0.01      ],
 [ 0.    ,      0.25881905,  0.96592583, -0.14      ],
 [ 0.    ,     -0.96592583,  0.25881905, -0.09     , ],
 [ 0.    ,      0.     ,     0.      ,    1.        ],],

 [[ 1.   ,       0.     ,     0.        , 0      ],
 [ 0.    ,      0.25881905,  0.96592583, -0.12      ],
 [ 0.    ,     -0.96592583,  0.21881905, 0  , ],
 [ 0.    ,      0.     ,     0.      ,    1.        ],],

 [[ 1.   ,       0.      ,    0.      ,    0.        ],
 [ 0.    ,      0.81915204 ,-0.57357644 , 0.12       ],
 [ 0.    ,      0.57357644,  0.81915204 , 0.        ],
 [ 0.     ,     0.       ,   0.         , 1.        ]],

[[ 1.     ,     0.     ,     0.        ,  0.        ],
 [ 0.    ,     -0.9961947,  0.08715574,  0.        ],
 [ 0.     ,    -0.08715574 ,-0.9961947 ,  0.2        ],
 [ 0.     ,     0.       ,   0.        ,  1.        ],],

 [[ 1.   ,      0.      ,   0.      ,   0.       ],
 [ 0.    ,    -0.5      ,  0.8660254,  -0.1       ],
 [ 0.    ,    -0.8660254, -0.5      ,  0.1       ],
 [ 0.    ,     0.     ,    0.       ,  1.       ],],

[[ 1.   ,       0.        , -0.,          0.        ],
 [-0.    ,      0.90630779, -0.42261826,  0.1        ],
 [ 0.     ,     0.42261826,  0.90630779 , -0.1        ],
 [ 0. ,         0.    ,      0.          ,1.        ],],

 [[ 1.    ,      0.       ,   0.  ,        0.        ],
 [ 0.     ,     0.17364818, -0.98480775 , 0.14        ],
 [ 0.    ,      0.98480775 , 0.17364818,  -0.03        ],
 [ 0.     ,     0.        ,  0.  ,        1.        ]]


 ]

grasp_scores = [0, 1, 0,1,1, 0, 0,0, 0,0]
for transform, score in zip(grasp_transforms,grasp_scores):
    gripper = utils.gripper_bd(score)
    gripper.apply_transform(transform)
    scene.add_geometry(gripper)


distance = 0.6
transform = np.eye(4)
r = R.from_euler('Y', 90, degrees=True).as_matrix()
r2 = R.from_euler('Z', 90, degrees=True).as_matrix()
transform[:3,:3] = r @ r2
point = [0,0,0]
new_transform = scene.camera.look_at(points=[point],distance=distance,rotation=transform)
scene.camera_transform = new_transform

# png = scene.save_image(resolution=[640*2, 480*2],
#                     visible=True)
# with open(f"sugar_box_bad", 'wb') as f:
#     f.write(png)
#     f.close()


scene.show()
# exit()
scene = Scene()
scene.add_geometry(obj_pcl)
# scene.show()

grasp_transforms = [
[[ 1,   0,   0.,   0., ],
 [ 0.  , 1. ,  0. ,  0. ],
 [ 0.  , 0. ,  1. , -0.17],
 [ 0.  , 0.  , 0.  , 1. ],],
 
 [[ 1,   0,   0.,   0., ],
 [ 0.  , 1. ,  0. ,  0.03 ],
 [ 0.  , 0. ,  1. , -0.17],
 [ 0.  , 0.  , 0.  , 1. ],],

 [[ 1.   ,       0.     ,     0.        , -0.01      ],
 [ 0.    ,      0.25881905,  0.96592583, -0.12      ],
 [ 0.    ,     -0.96592583,  0.25881905, -0.11     , ],
 [ 0.    ,      0.     ,     0.      ,    1.        ],],

 [[ 1.   ,       0.     ,     0.        , -0.01      ],
 [ 0.    ,      0.25881905,  0.96592583, -0.14      ],
 [ 0.    ,     -0.96592583,  0.25881905, -0.09     , ],
 [ 0.    ,      0.     ,     0.      ,    1.        ],],

 [[ 1.   ,       0.     ,     0.        , 0      ],
 [ 0.    ,      0.25881905,  0.96592583, -0.12      ],
 [ 0.    ,     -0.96592583,  0.21881905, 0  , ],
 [ 0.    ,      0.     ,     0.      ,    1.        ],],

 [[ 1.   ,       0.      ,    0.      ,    0.        ],
 [ 0.    ,      0.17364818,-0.98480775  , 0.12     ],
 [ 0.    ,       0.98480775,   0.17364818, 0.03       ],
 [ 0.     ,     0.       ,   0.         , 1.        ]],

[[ 1.     ,     0.     ,     0.        ,  0.        ],
 [ 0.    ,     -0.9961947,  0.08715574,  0.        ],
 [ 0.     ,    -0.08715574 ,-0.9961947 ,  0.16        ],
 [ 0.     ,     0.       ,   0.        ,  1.        ],],

 [[ 1.   ,      0.      ,   0.      ,   0.       ],
 [ 0.    ,    -0.5      ,  0.8660254,  -0.12       ],
 [ 0.    ,    -0.8660254, -0.5      ,  0.1       ],
 [ 0.    ,     0.     ,    0.       ,  1.       ],],

[[ 1.   ,       0.        , -0.,          0.        ],
 [-0.    ,      0.17364818, -0.98480775,  0.12        ],
 [ 0.     ,     0.98480775, 0.17364818 , -0.07        ],
 [ 0. ,         0.    ,      0.          ,1.        ],],

 [[ 1.    ,      0.       ,   0.  ,        0.        ],
 [ 0.     ,     0.17364818, -0.98480775 , 0.12        ],
 [ 0.    ,      0.98480775 , 0.17364818,  0.02       ],
 [ 0.     ,     0.        ,  0.  ,        1.        ]]
 ]

grasp_scores = [1, 1, 1,1,1, 1, 1,1, 1,1]
for transform, score in zip(grasp_transforms,grasp_scores):
    gripper = utils.gripper_bd(score)
    gripper.apply_transform(transform)
    scene.add_geometry(gripper)
scene.camera_transform = new_transform
scene.show()

scene = Scene()
scene.add_geometry(obj_pcl)
grasp_transforms = [
[[ 1,   0,   0.,   0., ],
 [ 0. ,   -0.9961947,  0.08715574,  -0.01 ],
 [ 0.  , -0.08715574 ,-0.9961947 , 0.16],
 [ 0.  , 0.  , 0.  , 1. ],],

[[ 1.  ,        0.  ,        0.  ,        0.  ,      ],
 [ 0.  ,       -0.9961947,  -0.08715575 , 0.03,      ],
 [ 0.  ,        0.08715575, -0.9961947 ,  0.16,      ],
 [ 0.  ,        0.  ,        0.  ,        1.  ,      ]],

 
 [[ 1.     ,     0.     ,     0.        ,  0.        ],
 [ 0.    ,     -0.9961947,  0.08715574,  0.        ],
 [ 0.     ,    -0.08715574 ,-0.9961947 ,  0.17        ],
 [ 0.     ,     0.       ,   0.        ,  1.        ],], 
 


 [[ 1.   ,      0.      ,   0.      ,   0.       ],
 [ 0.    ,    -0.8660254   , 0.5,  -0.07       ],
 [ 0.    ,    -0.5,         -0.8660254      ,  0.15       ],
 [ 0.    ,     0.     ,    0.       ,  1.       ],],

 [[ 1.   ,       0.     ,     0.        , 0      ],
 [ 0.    ,      -0.08715574,  0.9961947, -0.12      ],
 [ 0.    ,     -0.9961947,  -0.21881905, 0.08  , ],
 [ 0.    ,      0.     ,     0.      ,    1.        ],], 

 
 [[ 1.   ,       0.     ,     0.        , -0.01      ],
 [ 0.    ,      -0.08715574,  0.9961947, -0.12      ],
 [ 0.    ,     -0.9961947,  -0.08715574, 0.06    , ],
 [ 0.    ,      0.     ,     0.      ,    1.        ],], 
 
 [[ 1.   ,       0.     ,     0.        , -0.01      ],
 [ 0.    ,      -0.08715574,  0.9961947, -0.12      ],
 [ 0.    ,     -0.9961947,  -0.1,       0.035    , ],
 [ 0.    ,      0.     ,     0.      ,    1.        ],], 
 


  [[ 1.   ,       0.      ,    0.      ,    0.        ],
 [ 0.    ,      -0.18,-0.9961947  , 0.12     ],
 [ 0.    ,       0.9961947,   -0.18, 0.05       ],
 [ 0.     ,     0.       ,   0.         , 1.        ]],
 
 [[ 1.   ,       0.        , -0.,          0.        ],
 [-0.    ,      -0.08715574, -0.9961947,  0.12        ],
 [ 0.     ,     0.9961947, -0.08715574 , 0.034        ],
 [ 0. ,         0.    ,      0.          ,1.        ],], 
 
 [[ 1.    ,      0.       ,   0.  ,        0.        ],
 [ 0.     ,     -0.23, -0.9961947 , 0.12        ],
 [ 0.    ,      0.9961947 , -0.23,  0.085       ],
 [ 0.     ,     0.        ,  0.  ,        1.        ]] 
]

grasp_scores = [1, 1, 1,1,1, 1, 1,1, 1,1]
for transform, score in zip(grasp_transforms,grasp_scores):
    gripper = utils.gripper_bd(score)
    gripper2 = utils.gripper_bd(0)



    trans = np.array(transform)
    r2 = np.eye(4)
    r2[:3,:3] =  R.from_euler('xyz', [15, 0, 0], degrees=True).as_matrix()
    # print(r2)
    # exit()

    new = trans@r2
    print(new)
    gripper.apply_transform(transform)
    gripper2.apply_transform(new)
    scene.add_geometry(gripper)
    # scene.add_geometry(gripper2)
scene.camera_transform = new_transform
scene.show()
