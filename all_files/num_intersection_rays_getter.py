from math import degrees
import numpy as np
from graspsampler.common import PandaGripper, Scene, Object
import trimesh.transformations as tra
from graspsampler.GraspSampler import GraspSampler
from scipy.spatial.transform import Rotation as R
import threading
import json
import trimesh
import time 
import os
import pickle
from tqdm import tqdm

def get_num_intersection_rays(transforms, object_mesh, gripper):
    num_intersection_rays = []

    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            object_mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)
    
    collisions, _ = in_collision_with_gripper(object_mesh, transforms, gripper=gripper, silent=True)

    for transform, collision in zip(transforms,collisions):
        # if in collision, no need to retrieve the number of intersection rays
        if collision == 1:
            num_intersection_rays.append(0)
            continue 

        ray_origins, ray_directions = gripper.get_closing_rays(transform)
        locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins, ray_directions, multiple_hits=False)
        num_intersection_rays.append(len(locations))

    return np.array(num_intersection_rays)


def get_contact_points_and_ray_directions(transforms, object_mesh, gripper):
    contact_points = []
    force_directions = []

    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            object_mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)
        
    for transform in tqdm(transforms):
        ray_origins, ray_directions = gripper.get_closing_rays(transform)
        locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins, ray_directions, multiple_hits=False)
        if locations.size == 0:
            contact_points.append(np.zeros((2,3)))
            force_directions.append(np.zeros((2,3))) 
            continue
        index_ray_left = np.array([i for i, num in enumerate(
        index_ray) if num % 2 == 0 and np.linalg.norm(ray_origins[num]-locations[i]) < 2.0*gripper.q])
        index_ray_right = np.array([i for i, num in enumerate(
        index_ray) if num % 2 == 1 and np.linalg.norm(ray_origins[num]-locations[i]) < 2.0*gripper.q])
        if index_ray_left.size == 0 or index_ray_right.size == 0:
            contact_points.append(np.zeros((2,3)))
            force_directions.append(np.zeros((2,3))) 
            continue
        left_contact_idx = np.linalg.norm(
                            ray_origins[index_ray[index_ray_left]] - locations[index_ray_left], axis=1).argmin()
        right_contact_idx = np.linalg.norm(
                            ray_origins[index_ray[index_ray_right]] - locations[index_ray_right], axis=1).argmin()
   
        left_contact_point = locations[index_ray_left[left_contact_idx]]
        right_contact_point = locations[index_ray_right[right_contact_idx]]
        contact_points.append([left_contact_point,right_contact_point])
        force_directions.append([ray_directions[0],ray_directions[1]])
    return np.asarray(contact_points), np.asarray(force_directions)
    
def get_transforms(grasps_data):
    translations = grasps_data["translations"]
    quaternions = grasps_data["quaternions"]
    transforms = []
    for t, q in zip(translations, quaternions):
        transform = np.eye(4)
        r = R.from_quat(q)
        transform[:3,:3] = r.as_matrix()
        transform[:3,3] = t
        transforms.append(transform)
    return np.asarray(transforms)

def in_collision_with_gripper(object_mesh, gripper_transforms, gripper, silent=False):
    
    """Check collision of object with gripper.
    Arguments:
        object_mesh {trimesh} -- mesh of object
        gripper_transforms {list of numpy.array} -- homogeneous matrices of gripper
        gripper_name {str} -- name of gripper
    Keyword Arguments:
        silent {bool} -- verbosity (default: {False})
    Returns:
        [list of bool] -- Which gripper poses are in collision with object mesh
    """
    manager = trimesh.collision.CollisionManager()
    manager.add_object('object', object_mesh)
    gripper_meshes = gripper.get_meshes()
    min_distance = []
    for tf in tqdm(gripper_transforms):
        min_distance.append(np.min([manager.min_distance_single(
            gripper_mesh, transform=tf) for gripper_mesh in gripper_meshes]))
    return np.asarray([d == 0 for d in min_distance]), min_distance

def get_force_closure_labels_file(cat, idx, grasp_data_filename):
    grasp_data = np.load(grasp_data_filename)
    for key in grasp_data:
        print(key)

    ori_grasp_data_filename = f"/home/user/isaac/grasp_data_generated/{cat}/{cat}{idx:03}/main1.npz"
    _ori_grasp_data = np.load(ori_grasp_data_filename)
    grasp_transforms = get_transforms(grasp_data)
    obj_pose_relative = _ori_grasp_data["obj_pose_relative"]
    obj_filename = f"grasp_data/meshes/{cat}/{cat}{idx:03}.stl" if cat in ["box","cylinder"] else  f"grasp_data/meshes/{cat}/{cat}{idx:03}.obj"
    metadata_filename = f'grasp_data/info/{cat}/{cat}{idx:03}.json'
    metadata = json.load(open(metadata_filename,'r'))
    sampler = GraspSampler()
    sampler.update_object(obj_filename=obj_filename, name=obj_filename, obj_scale=metadata['scale'])
    grasp_collisions, _ = in_collision_with_gripper(sampler.obj.mesh, grasp_transforms, gripper=sampler.gripper)
    contact_points = np.zeros((len(grasp_collisions),2,3))
    ray_directions = np.zeros((len(grasp_collisions),2,3))
    mask = np.argwhere(grasp_collisions==False).flatten()
    contact_points_mask, ray_directions_mask = get_contact_points_and_ray_directions(grasp_transforms[mask], sampler.obj.mesh, sampler.gripper)
    print(contact_points_mask.shape)
    contact_points_mask[:,:] += obj_pose_relative
    contact_points[mask] = contact_points_mask
    ray_directions[mask] = ray_directions_mask
    save_dir = f"force_closure_data/{cat}/{cat}{idx:03}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_name = f"force_closure_data/{cat}/{cat}{idx:03}/" + grasp_data_filename.split("/")[-1]
    np.savez_compressed(file=file_name,quaternions=grasp_data["quaternions"], translations=grasp_data["translations"],
    contact_points=contact_points,ray_directions=ray_directions,isaac_labels=grasp_data["isaac_labels"])

def get_force_closure_labels_object(cat, idx, trial):
    main_grasps_data_filename = f"/home/user/isaac/grasp_data_generated/{cat}/{cat}{idx:03}_isaac/main{trial}.npz"
    rest_grasps_data_filename = f"/home/user/isaac/grasp_data_generated/{cat}/{cat}{idx:03}_isaac/main{trial}_rest.npz"
    get_force_closure_labels_file(cat,idx,main_grasps_data_filename)
    get_force_closure_labels_file(cat,idx,rest_grasps_data_filename)

def get_num_intersection_rays_file(cat,idx):

    data_path = f"/home/user/isaac/grasps_sampled_for_positive_grasps/{cat}/{cat}{idx:03}_isaac"
    save_dir = f"/home/user/isaac/grasps_sampled_for_positive_grasps/{cat}/{cat}{idx:03}_isaac_lower_friction_with_num_intersection_rays"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    obj_filename = f"grasp_data/meshes/{cat}/{cat}{idx:03}.stl" if cat in ["box","cylinder"] else  f"grasp_data/meshes/{cat}/{cat}{idx:03}.obj"
    metadata_filename = f'grasp_data/info/{cat}/{cat}{idx:03}.json'
    metadata = json.load(open(metadata_filename,'r'))
    sampler = GraspSampler()
    sampler.update_object(obj_filename=obj_filename, name=obj_filename, obj_scale=metadata['scale'])

    for file_idx in range(999,1000):
        grasp_data_filename = f"{data_path}/{file_idx:08}.npz"
        grasp_data = np.load(grasp_data_filename)
        for key in grasp_data:
            print(key)
        grasp_transforms = get_transforms(grasp_data)
        num_intersection_rays = get_num_intersection_rays(grasp_transforms,  sampler.obj.mesh, sampler.gripper)
        file_name = f"{save_dir}/{file_idx:08}.npz"
        np.savez_compressed(file=file_name,quaternions=grasp_data["quaternions"], translations=grasp_data["translations"],
        isaac_labels=grasp_data["isaac_labels"],num_intersection_rays=num_intersection_rays)

cat = "bowl" #"cylinder" box bowl mug

for idx in range(1):
    # try:
        get_num_intersection_rays_file(cat,idx)
    # except:
    #     continue
# main_grasps_data_filename = f"/home/user/isaac/grasp_data_generated/{cat}/{cat}{idx:03}_isaac/main{trial}.npz"
# rest_grasps_data_filename = f"/home/user/isaac/grasp_data_generated/{cat}/{cat}{idx:03}_isaac/main{trial}_rest.npz"
# ori_grasp_data_filename = f"/home/user/isaac/grasp_data_generated/{cat}/{cat}{idx:03}/main{trial}.npz"
# _main_grasps_data = np.load(main_grasps_data_filename)
# _rest_grasps_data = np.load(rest_grasps_data_filename)
# _ori_grasp_data = np.load(ori_grasp_data_filename)
# obj_pose_relative = _ori_grasp_data["obj_pose_relative"]

# main_grasp_transforms = get_transforms(_main_grasps_data)
# rest_grasp_transforms = get_transforms(_rest_grasps_data)

# obj_filename = f"grasp_data/meshes/{cat}/{cat}{idx:03}.stl" if cat in ["box","cylinder"] else  f"grasp_data/meshes/{cat}/{cat}{idx:03}.obj"
# metadata_filename = f'grasp_data/info/{cat}/{cat}{idx:03}.json'
# metadata = json.load(open(metadata_filename,'r'))
# print("scale", metadata['scale'])

# sampler = GraspSampler()
# sampler.update_object(obj_filename=obj_filename, name=obj_filename, obj_scale=metadata['scale'])
# main_grasp_collisions, _ = in_collision_with_gripper(sampler.obj.mesh, main_grasp_transforms, gripper=sampler.gripper)
# contact_points = np.zeros((len(main_grasp_collisions),2,3))
# ray_directions = np.array([[[0,0,0],[0,0,0]] for _ in range(len(main_grasp_collisions))])
# mask = np.argwhere(main_grasp_collisions==False).flatten()
# contact_points_mask, ray_directions_mask = get_contact_points_and_ray_directions(main_grasp_transforms[mask], sampler.obj.mesh, sampler.gripper)
# contact_points_mask[:,:] += obj_pose_relative
# contact_points[mask] = contact_points_mask
# ray_directions[mask] = ray_directions_mask



# save_dir = f"force_closure_data/{cat}/{cat}{idx:03}"
# if not os.path.exists(save_dir):
# 	os.makedirs(save_dir)





# print(contact_points_mask.shape, ray_directions_mask.shape)
# for transform in main_grasp_transforms[mask]:
#     # print(contact_point[0], ray_direction[0])
#     # sampler = GraspSampler()
#     # sampler.update_object(obj_filename=obj_filename, name=obj_filename, obj_scale=metadata['scale'])
#     contact_points, ray_directions = get_contact_points_and_ray_directions([transform], sampler.obj.mesh, sampler.gripper)
#     sampler.scene.add_rays((contact_points[0], ray_directions[0]))
#     sampler.grasp_visualize(transform, coordinate_frame=True,grasp_debug_items=False,other_debug_items=False)
