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


obj_path = "assets/meshes_10_objects/048_hammer/google_16k/textured.obj"
obj_mesh = Object(filename=obj_path, name="sugar_box")
obj_mesh = obj_mesh.mesh

point_idx = np.random.choice(len(obj_mesh.vertices),4000,replace=False)
points = obj_mesh.vertices[point_idx]
points_color = obj_mesh.visual.to_color().vertex_colors[point_idx]
obj_pcl = trimesh.points.PointCloud(vertices=points, colors=points_color)

scene = Scene()
scene.add_geometry(obj_pcl)
# scene.show()

grasp_transforms = [
[[-0.5      ,  0.      ,   0.8660254 , -0.06       ],
 [ 0.       ,  1.      ,   0.        , 0.       ],
 [-0.8660254 , 0.      ,  -0.5        ,0.04       ],
 [ 0.      ,   0.      ,   0.        , 1.       ],]
,

 
[[-0.5      ,  0.      ,   0.8660254 , -0.05       ],
 [ 0.       ,  1.      ,   0.        , -0.1       ],
 [-0.8660254 , 0.      ,  -0.5        ,0.06       ],
 [ 0.      ,   0.      ,   0.        , 1.       ],]
,


[[-0.08715574, -0.    ,     -0.9961947  , 0.11        ],
 [ 0.        ,  1.    ,     -0.         , -0.02        ],
 [ 0.9961947 ,  0.    ,     -0.08715574 , 0.02        ],
 [ 0.        ,  0.    ,      0.         , 1.        ],]
,

[[-0.08715574, -0.    ,     -0.9961947  , 0.15        ],
 [ 0.        ,  1.    ,     -0.         , -0.06        ],
 [ 0.9961947 ,  0.    ,     -0.08715574 , 0.02        ],
 [ 0.        ,  0.    ,      0.         , 1.        ],]
,

[[-0.07547909, -0.04357787, -0.9961947  , 0.17 +0.02     ],
 [-0.5       ,  0.8660254 ,  0.         , -0.07 -0.06      ],
 [ 0.86272992,  0.49809735, -0.08715574 , 0.02          ],
 [ 0.        ,  0.        ,  0.         , 1.        ],]
,

[[-8.71557427e-02, -9.96194698e-01,  5.55111512e-17,  0.00000000e+00],
 [ 8.68240888e-02, -7.59612349e-03, -9.96194698e-01,  0.16],
 [ 9.92403877e-01, -8.68240888e-02,  8.71557427e-02,  0.00000000e+00],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],]
,

[[ 2.77555756e-17, -8.66025404e-01,  5.00000000e-01 , -0.1],
 [ 8.71557427e-02 ,-4.98097349e-01, -8.62729916e-01 , 0.19],
 [ 9.96194698e-01  ,4.35778714e-02,  7.54790873e-02 , 0.00000000e+00],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]]
,

[[-0.5      ,  0.      ,   0.8660254 , -0.06       ],
 [ 0.       ,  1.      ,   0.        , -0.16      ],
 [-0.8660254 , 0.      ,  -0.5        ,0.04       ],
 [ 0.      ,   0.      ,   0.        , 1.       ],]
,

[[-0.64034161,  0.76312941, -0.08715574 , 0.06        ],
 [ 0.20935858,  0.28258505,  0.93611681 , -0.29        ],
 [ 0.73900718,  0.58118774, -0.34071865 , 0.        ],
 [ 0.        ,  0.        ,  0.         , 1.        ]],

 [[-0.28678822, -0.40957602 , 0.8660254,   -0.17       ],
 [-0.18660867, -0.86279873, -0.46984631,  0.11        ],
 [ 0.9396434 , -0.29635424 , 0.17101007,  0.        ],
 [ 0.        ,  0.         , 0.        ,  1.        ]]
,

 ]

grasp_scores = [0, 1, 1,0,0, 0, 0,0, 0,0]
for transform, score in zip(grasp_transforms,grasp_scores):
    gripper = utils.gripper_bd(score)
    gripper.apply_transform(transform)
    scene.add_geometry(gripper)


distance = 0.9
transform = np.eye(4)
r = R.from_euler('Y', 0, degrees=True).as_matrix()
r2 = R.from_euler('Z', 22, degrees=True).as_matrix()
transform[:3,:3] = r @ r2
point = [0,0,-0.09]
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
scene.show()

grasp_transforms = [
[[-0.5      ,  0.      ,   0.8660254 , -0.10      ],
 [ 0.       ,  1.      ,   0.        , 0.       ],
 [-0.8660254 , 0.      ,  -0.5        ,0.08       ],
 [ 0.      ,   0.      ,   0.        , 1.       ],]
,

 
[[-0.5      ,  0.      ,   0.8660254 , -0.05       ],
 [ 0.       ,  1.      ,   0.        , -0.1       ],
 [-0.8660254 , 0.      ,  -0.5        ,0.06       ],
 [ 0.      ,   0.      ,   0.        , 1.       ],]
,


[[-0.08715574, -0.    ,     -0.9961947  , 0.11        ],
 [ 0.        ,  1.    ,     -0.         , -0.02        ],
 [ 0.9961947 ,  0.    ,     -0.08715574 , 0.02        ],
 [ 0.        ,  0.    ,      0.         , 1.        ],]
,

[[-0.08715574, -0.    ,     -0.9961947  , 0.12        ],
 [ 0.        ,  1.    ,     -0.         , -0.06        ],
 [ 0.9961947 ,  0.    ,     -0.08715574 , 0.02        ],
 [ 0.        ,  0.    ,      0.         , 1.        ],]
,

[[-0.07547909, -0.04357787, -0.9961947  , 0.15    ],
 [-0.5       ,  0.8660254 ,  0.         , -0.07 -0.06      ],
 [ 0.86272992,  0.49809735, -0.08715574 , 0.02          ],
 [ 0.        ,  0.        ,  0.         , 1.        ],]
,

[[-8.71557427e-02, -9.96194698e-01,  5.55111512e-17,  0.00000000e+00],
 [ 8.68240888e-02, -7.59612349e-03, -9.96194698e-01,  0.18],
 [ 9.92403877e-01, -8.68240888e-02,  8.71557427e-02,  0.00000000e+00],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],]
,

[[ 2.77555756e-17, -8.66025404e-01,  5.00000000e-01 , -0.08],
 [ 8.71557427e-02 ,-4.98097349e-01, -8.62729916e-01 , 0.16],
 [ 9.96194698e-01  ,4.35778714e-02,  7.54790873e-02 , 0.00000000e+00],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]]
,

[[-0.5      ,  0.      ,   0.8660254 , -0.03       ],
 [ 0.       ,  1.      ,   0.        , -0.16      ],
 [-0.8660254 , 0.      ,  -0.5        ,0.04       ],
 [ 0.      ,   0.      ,   0.        , 1.       ],]
,

[[-0.64034161,  0.76312941, -0.08715574 , 0.095       ],
 [ 0.20935858,  0.28258505,  0.93611681 , -0.29        ],
 [ 0.73900718,  0.58118774, -0.34071865 , 0.03        ],
 [ 0.        ,  0.        ,  0.         , 1.        ]],

 [[-0.28678822, -0.40957602 , 0.8660254,   -0.13     ],
 [-0.18660867, -0.86279873, -0.46984631,  0.11        ],
 [ 0.9396434 , -0.29635424 , 0.17101007,  0.        ],
 [ 0.        ,  0.         , 0.        ,  1.        ]]
,

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
[[-0.5      ,  0.      ,   0.8660254 , -0.10      ],
 [ 0.       ,  1.      ,   0.        , 0.       ],
 [-0.8660254 , 0.      ,  -0.5        ,0.08       ],
 [ 0.      ,   0.      ,   0.        , 1.       ],]
,

 
[[-0.5      ,  0.      ,   0.8660254 , -0.05       ],
 [ 0.       ,  1.      ,   0.        , -0.04       ],
 [-0.8660254 , 0.      ,  -0.5        ,0.06       ],
 [ 0.      ,   0.      ,   0.        , 1.       ],]
,


[[-0.08715574, -0.    ,     -0.9961947  , 0.08        ],
 [ 0.        ,  1.    ,     -0.         , 0.02        ],
 [ 0.9961947 ,  0.    ,     -0.08715574 , 0.02        ],
 [ 0.        ,  0.    ,      0.         , 1.        ],]
,

[[-0.08715574, -0.    ,     -0.9961947  , 0.07        ],
 [ 0.        ,  1.    ,     -0.         , 0.05        ],
 [ 0.9961947 ,  0.    ,     -0.08715574 , 0.02        ],
 [ 0.        ,  0.    ,      0.         , 1.        ],]
,

[[-0.07547909, -0.04357787, -0.9961947  , 0.07    ],
 [-0.5       ,  0.8660254 ,  0.         , 0.06      ],
 [ 0.86272992,  0.49809735, -0.08715574 , 0.02          ],
 [ 0.        ,  0.        ,  0.         , 1.        ],]
,

[[-8.71557427e-02, -9.96194698e-01,  5.55111512e-17,  0.00000000e+00],
 [ 8.68240888e-02, -7.59612349e-03, -9.96194698e-01,  0.18],
 [ 9.92403877e-01, -8.68240888e-02,  8.71557427e-02,  0.02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],]
,

[[ 2.77555756e-17, -8.66025404e-01,  5.00000000e-01 , -0.08],
 [ 8.71557427e-02 ,-4.98097349e-01, -8.62729916e-01 , 0.16],
 [ 9.96194698e-01  ,4.35778714e-02,  7.54790873e-02 , 0.00000000e+00],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]]
,

[[-0.5      ,  0.      ,   0.8660254 , -0.09       ],
 [ 0.       ,  1.      ,   0.        , 0.01     ],
 [-0.8660254 , 0.      ,  -0.5        ,0.04       ],
 [ 0.      ,   0.      ,   0.        , 1.       ],]
,

[[-0.07547909, -0.04357787, -0.9961947  , 0.10    ],
 [-0.5       ,  0.8660254 ,  0.         , 0.00      ],
 [ 0.86272992,  0.49809735, -0.08715574 , 0.02          ],
 [ 0.        ,  0.        ,  0.         , 1.        ],]
,

 [[-0.28678822, -0.40957602 , 0.8660254,   -0.13     ],
 [-0.18660867, -0.86279873, -0.46984631,  0.11        ],
 [ 0.9396434 , -0.29635424 , 0.17101007,  0.        ],
 [ 0.        ,  0.         , 0.        ,  1.        ]]
,

 ]

grasp_scores = [1,1,1, 1,1,1,1, 1,1, 1,1]
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
