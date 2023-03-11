from trimesh.permutate import transform
from graspsampler.common import PandaGripper, Scene, Object
import trimesh
import pickle
import numpy as np
from graspsampler.PointCloudManager import PointCloudManager
from graspsampler.utils import regularize_pc_point_count

# scene = Scene()


# gripper = PandaGripper(root_folder='')
# gripper_full_mesh = trimesh.util.concatenate( gripper.get_meshes() )
# # gripper_full_mesh.visual.face_colors = [255,0,0,100]
# pc_manager = PointCloudManager(seed=10)

# pc_manager.update_object(obj_mesh=gripper_full_mesh)
# # pc_manager.set_obj_pose(np.eye(4))

# # _, _, pc, transferred_pose = pc_manager.render_pc(full_pc=True)
# _, _, pcs = pc_manager.render_multiple_pcs(number_of_pcs=10, target_pc_size=1024, depth_noise=0, dropout=0)


# # for pc in pcs:
# #     pc_mesh = trimesh.points.PointCloud(pc)
# #     scene.add_geometry(pc_mesh)
# pc = np.concatenate(pcs, axis=0)
# print(f'pc shape: {pc.shape}')
# pc = regularize_pc_point_count(pc, 1024, True)
# print(f'pc shape: {pc.shape}')

# pc_mesh = trimesh.points.PointCloud(pc)
# scene.add_geometry(pc_mesh)

# scene.add_gripper(gripper)
# scene.add_coordinate_frame()


# # scene.add_geometry(gripper_full_mesh)
# scene.show()

# # np.save('assets/gripper_control_points/full_gripper_pc', pc, allow_pickle=False)


gripper = PandaGripper(root_folder='')

control_points = np.load('assets/gripper_control_points/panda.npy')

scene = Scene()
scene.add_gripper(gripper)
# scene.add_coordinate_frame()
scene.show()

# print(control_points)
def draw_point(z):
    sp = trimesh.primitives.Sphere(radius=0.001, center=z)
    scene.add_geometry(sp)

new_control_points = []

p1 = control_points[0][:3]
p1[0] -= 0.0025
p1[2] -= 0.012
draw_point(p1)
new_control_points.append(p1)

p2 = control_points[1][:3]
p2[0] += 0.0025
p2[2] -= 0.012
draw_point(p2)
new_control_points.append(p2)

for i in range(1,9,1):
    _p = np.copy(p1)
    _p[0] = (p1[0] - p2[0])*i/9.0 + p2[0]
    draw_point(_p)
    new_control_points.append(_p)

p3 = control_points[18][:3]
p3[0] -= 0.0025
p3[2] += 0.005
draw_point(p3)

for i in range(1,10,1):
    _p = np.copy(p1)
    _p[2] = (p3[2] - p1[2])*i/10.0 + p1[2]
    draw_point(_p)
    new_control_points.append(_p)

p4 = control_points[19][:3]
p4[0] += 0.0025
p4[2] += 0.005
draw_point(p4)

for i in range(1,10,1):
    _p = np.copy(p2)
    _p[2] = (p4[2] - p2[2])*i/10.0 + p2[2]
    draw_point(_p)
    new_control_points.append(_p)

new_control_points = np.array(new_control_points)
print(new_control_points.shape)

# np.save('assets/gripper_control_points/new_control_points', new_control_points, allow_pickle=False)
scene.show()

