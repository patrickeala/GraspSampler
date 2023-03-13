import numpy as np

import pickle
from graspsampler.GraspSampler import GraspSampler
from graspsampler.common import PandaGripper, Scene, Object
import trimesh
from graspsampler.utils import regularize_pc_point_count

graspsampler = GraspSampler(seed=10)

i = 10
obj_filename = f'assets/sample_files/box{i:03}.stl'

NUM_PCS = 30
TARGET_PC_SIZE  = 1024
graspsampler.update_object(obj_filename=obj_filename, name=obj_filename)

pcs = graspsampler.get_multiple_random_pcs(number_of_pcs=NUM_PCS, 
                                            target_pc_size=TARGET_PC_SIZE, 
                                            depth_noise=0.00, 
                                            dropout=0.0)


scene = Scene()
# scene.add_object(graspsampler.obj)

for k in range(NUM_PCS):
    pc_obj = trimesh.points.PointCloud(pcs[k])
    scene.add_geometry(pc_obj)
    

scene.show()

# combine all point clouds


pc = np.concatenate(pcs)
print(pc.shape)

pc = regularize_pc_point_count(pc, TARGET_PC_SIZE, True)

print(pc.shape)

np.save(f'box{i:03}', pc)

scene = Scene()

pc_obj = trimesh.points.PointCloud(pc)
scene.add_geometry(pc_obj)
    

scene.show()