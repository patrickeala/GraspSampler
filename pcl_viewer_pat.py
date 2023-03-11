from numpy.random.mtrand import rand
from trimesh.permutate import transform
from graspsampler.common import PandaGripper, Scene, Object
from graspsampler.GraspSampler import GraspSampler
from graspsampler.utils import gripper_bd
import trimesh
import pickle
import numpy as np

from scipy.spatial.transform import Rotation as R


# def read_pc(cat, id, view_size=4):

#     with open(f'data/pcs/{cat}/{cat}{id:03}.pkl', 'rb') as fp:
#         data = pickle.load(fp)
#         obj_pose_relative = data['obj_pose_relative']
#         pcs = data['pcs']
    
#     pc_indcs = np.random.randint(low=0, high=999, size=view_size)
    
#     if len(pc_indcs) == 1:
#         pc = pcs[pc_indcs[0]]
#     else:
#         __pcs = []
#         for pc_index in pc_indcs:
#             __pcs = __pcs + [pcs[pc_index]]
#         pc = np.concatenate( __pcs )
   
#     pc = regularize_pc_point_count(pc, 1024, False)
#     return pc, obj_pose_relative


# with open('004_sugar_box.pkl',"rb") as handle:
    # pcs = np.array(pickle.load(handle))
    # print(pcs[0].shape)
file = "/home/gpupc2/GRASP/grasp_network/data/pcs/fork/fork001.pkl"
with open(file,"rb") as handle:
    # pc = np.array(pickle.load(handle))
    data = pickle.load(handle)
    pcs = data['pcs']
    print(len(pcs))
    # print(pcs[0].shape)

pcs_ = np.concatenate(pcs[:5])

pc1 = trimesh.points.PointCloud(vertices=pcs_)
scene = Scene()
scene.add_geometry(pc1)
scene.show()


def read_pc(cat, id, view_size=4):

    with open(f'data/pcs/{cat}/{cat}{id:03}.pkl', 'rb') as fp:
        data = pickle.load(fp)
        obj_pose_relative = data['obj_pose_relative']
        pcs = data['pcs']
    
    pc_indcs = np.random.randint(low=0, high=999, size=view_size)
    
    if len(pc_indcs) == 1:
        pc = pcs[pc_indcs[0]]
    else:
        __pcs = []
        for pc_index in pc_indcs:
            __pcs = __pcs + [pcs[pc_index]]
        pc = np.concatenate( __pcs )
   
    pc = regularize_pc_point_count(pc, 1024, False)
    return pc, obj_pose_relative
