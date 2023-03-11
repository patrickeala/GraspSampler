from numpy.random.mtrand import rand
from trimesh.permutate import transform
from graspsampler.common import PandaGripper, Scene, Object
from graspsampler.GraspSampler import GraspSampler
from graspsampler.utils import gripper_bd
import trimesh
import pickle
import numpy as np

from scipy.spatial.transform import Rotation as R

with open('004_sugar_box.pkl',"rb") as handle:
    pcs = np.array(pickle.load(handle))
    print(pcs[0].shape)
with open('004_sugar_box_origin.pkl',"rb") as handle:
    pc = np.array(pickle.load(handle))
    pc[0] = pc[0,:] + 0.1
pcs_ = pc[0]
print(pcs_.shape)
for _pc in pcs:
    pcs_ = np.concatenate([pcs_,_pc])
pc1 = trimesh.points.PointCloud(vertices=pcs_)
scene = Scene()
scene.add_geometry(pc1)
scene.show()
