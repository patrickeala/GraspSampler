import numpy as np
from trimesh.permutate import transform
from trimesh.visual import color
from utils.cad import PandaGripper, Object, grasp_quality_antipodal, create_gripper, grasp_quality_point_contacts, in_collision_with_gripper, sample_multiple_grasps, in_collision_with_gripper_test #, depth_to_pointcloud
import trimesh
from tqdm import tqdm
import trimesh.transformations as tra
import copy

import pyrender
import matplotlib.pyplot as plt
import time
import os




# obj_name = "025_mug"
obj_name = "004_sugar_box"
filename = f"/home/patrick/Desktop/grasper/assets/meshes_10_objects/{obj_name}/google_16k/textured.obj"
category = "box"

obj = Object(filename, category)
# obj = Object('assets/sample_files/box000.stl', 'box')


while True:
    root_folder = ''
    gripper = PandaGripper(root_folder=root_folder, face_colors=[125,125,125,100])

    big_scene = gripper.get_assemble_scene(add_coordinate_system=True)
    big_scene.add_geometry(obj.mesh, geom_name='object')
    # big_scene.show()

    ray_visualize = trimesh.load_path(np.hstack((gripper.ray_origins[:,:3],
                                            #  gripper.ray_origins[:,:3] + gripper.ray_directions*0.05)).reshape(-1, 2, 3))
                                             gripper.ray_origins[:,:3] + gripper.ray_directions*0.1)).reshape(-1, 2, 3))
    number_of_candidates = 1
    points, face_indices = obj.mesh.sample(
            number_of_candidates, return_index=True)
    normals = obj.mesh.face_normals[face_indices]
    angle = np.random.rand() * 2 * np.pi
    standoff = (gripper.standoff_range[1] - gripper.standoff_range[0]) * np.random.rand() \
            + gripper.standoff_range[0]

    # sampled point
    # _c = trimesh.primitives.Sphere(radius=0.005, center=points[0])
    # _c.visual.face_colors=[200,200,200,255]
    # big_scene.add_geometry(_c)

    # gripper location
    origin = points[0] + normals[0] * standoff
    # _o = trimesh.primitives.Sphere(radius=0.005, center=origin)
    # _o.visual.face_colors=[50,50,50,255]
    # big_scene.add_geometry(_o)

    # orientation
    orientation = tra.quaternion_matrix(
            tra.quaternion_about_axis(angle, [0, 0, 1]))
    tf = np.dot(np.dot(tra.translation_matrix(origin),
                            trimesh.geometry.align_vectors([0, 0, -1], normals[0])),
                    orientation)

    # apply gripper transform
    for _mesh in gripper.get_meshes():
            _mesh.visual.face_colors = [200,200,200,50]
            big_scene.add_geometry(_mesh.apply_transform(tf))

    # print(f"orientation: {orientation}")




    for _ray_origin in gripper.ray_origins:
        _ray_mesh = trimesh.primitives.Sphere(radius=0.001, center=_ray_origin[:3])
        _ray_mesh.visual.face_colors = [255, 255, 0, 255]
        #print(_ray_mesh)
        big_scene.add_geometry(_ray_mesh.apply_transform(tf))
    big_scene.add_geometry(ray_visualize.apply_transform(tf), geom_name='ray')




   
    collisions,_ = in_collision_with_gripper(
            obj.mesh, [tf], gripper_name='panda', silent=False)
    # collisions = in_collision_with_gripper_test(
            # obj.mesh, [tf], gripper_name='panda', silent=True)
    quality, locations = grasp_quality_antipodal(
            [tf], collisions, object_mesh=obj.mesh, gripper_name='panda', silent=False)
    quality2 = grasp_quality_point_contacts(
            [tf], collisions, object_mesh=obj.mesh, gripper_name='panda', silent=False)


    if len(locations) > 0 :
        for loc in locations:
            # sampled point
            _c = trimesh.primitives.Sphere(radius=0.0025, center=loc)
            _c.visual.face_colors=[200,200,200,255]
            big_scene.add_geometry(_c)

    print(f'Quality: {quality} and {quality2}')
    # big_scene.show()
    if quality[0] > 0.7:
    # if quality2[0] > 0:
        big_scene.show()

