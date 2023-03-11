from pickle import POP_MARK
from turtle import distance, shape
from typing import final
import numpy as np
from numpy.linalg.linalg import _multidot_dispatcher
from numpy.random.mtrand import normal
import trimesh.transformations as tra
from tqdm import tqdm
import trimesh
from scipy.spatial.transform import Rotation as R
from numpy import genfromtxt
import itertools

def trans_matrix(euler=[0, np.pi/4, np.pi/2], translation=[0.3,0.0,0.35]):
    """
    Builds homogenous transformation matrix from euler angles and translations.
    Args:
        * ``euler`` ((3,) ``float``): Euler angles in ``xyz`` order.
        * ``translation`` ((3,1) ``float``): Translations
    Returns:
        * ``H`` ((4,4) ``float``): Homogenous transformation matrix.
    """
    H = tra.euler_matrix(euler[0],euler[1],euler[2], 'rxyz')
    H[:3,3] = translation
    return H

def grasp_quality_weighted_antipodal(transforms, collisions, object_mesh, gripper, silent=False):
    """Grasp quality function
    Arguments:
        transforms {[type]} -- grasp poses
        collisions {[type]} -- collision information
        object_mesh {trimesh} -- object mesh
    Keyword Arguments:
        gripper_name {str} -- name of gripper (default: {'panda'})
        silent {bool} -- verbosity (default: {False})
    Returns:
        list of float -- quality of grasps [0..1]
    """
    res = []
    # gripper = create_gripper(gripper_name)
    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            object_mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)
    for p, colliding in tqdm(zip(transforms, collisions), total=len(transforms), disable=silent):
        if colliding:
            res.append(0)
        else:
            ray_origins, ray_directions = gripper.get_closing_rays(p)
            locations, index_ray, index_tri = intersector.intersects_location(
                ray_origins, ray_directions, multiple_hits=False)

            if len(locations) == 0:
                res.append(0)
            else:
                # this depends on the width of the gripper
                valid_locations = np.linalg.norm(
                    ray_origins[index_ray]-locations, axis=1) < 2.0*gripper.q

                if sum(valid_locations) == 0:
                    res.append(0)
                else:
                    res.append(1)
                    #contact_normals = object_mesh.face_normals[index_tri[valid_locations]]
                    #motion_normals = ray_directions[index_ray[valid_locations]]
                    #dot_prods = -(motion_normals * contact_normals).sum(axis=1)
                    #_res = np.abs(dot_prods.sum() / len(ray_origins))
                    #res.append(_res)
    return res


def get_contact_points_and_ray_directions(transforms, object_mesh, gripper):
    contact_points = []
    ray_directions = []

    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            object_mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)
        
    for transform in transforms:
        ray_origins, ray_directions = gripper.get_closing_rays(transform)
        locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins, ray_directions, multiple_hits=False)
        if locations.size == 0:
            contact_points.append(None)
            ray_directions.append(None) 
            continue
        index_ray_left = np.array([i for i, num in enumerate(
        index_ray) if num % 2 == 0 and np.linalg.norm(ray_origins[num]-locations[i]) < 2.0*gripper.q])
        index_ray_right = np.array([i for i, num in enumerate(
        index_ray) if num % 2 == 1 and np.linalg.norm(ray_origins[num]-locations[i]) < 2.0*gripper.q])
        left_contact_idx = np.linalg.norm(
                            ray_origins[index_ray[index_ray_left]] - locations[index_ray_left], axis=1).argmin()
        right_contact_idx = np.linalg.norm(
                            ray_origins[index_ray[index_ray_right]] - locations[index_ray_right], axis=1).argmin()
        left_contact_point = locations[index_ray_left[left_contact_idx]]
        right_contact_point = locations[index_ray_right[right_contact_idx]]
        contact_points.append([left_contact_point,right_contact_point])
        ray_directions.append([ray_directions[0],ray_directions[1]])
    return contact_points, ray_directions

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
        num_intersection_rays.append(locations.size)

    return num_intersection_rays


def get_minimum_gripper_distances_to_object(transforms,object_mesh,gripper):
    distances = []

    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            object_mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)


    for transform in transforms:
        ray_origins, ray_directions = gripper.get_transformed_get_distance_rays(transform)

        locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins, ray_directions, multiple_hits=False)
        print(locations.shape)
        if len(locations) == 0:
            distances.append(-1) # if no intersection, then distance is -1
            continue
        cur_distances = np.linalg.norm(ray_origins[index_ray]-locations,axis=1)
        min_cur_distance = min(cur_distances)
        distances.append(min_cur_distance)

    return distances


def sample_multiple_grasps(mesh, gripper,
                            number_of_grasps=1000,
                            alpha_lim = [0, 2*np.pi],
                            beta_lim = [0, 0],
                            gamma_lim = [0, 0],
                            silent=False):
    """Sample a set of grasps for an object.
    Arguments:
        number_of_candidates {int} -- Number of grasps to sample
        mesh {trimesh} -- Object mesh
        gripper_name {str} -- Name of gripper model
        systematic_sampling {bool} -- Whether to use grid sampling for roll
    Keyword Arguments:
        surface_density {float} -- surface density, in m^2 (default: {0.005*0.005})
        standoff_density {float} -- density for standoff, in m (default: {0.01})
        roll_density {float} -- roll density, in deg (default: {15})
        quality_type {str} -- quality metric (default: {'antipodal'})
        min_quality {float} -- minimum grasp quality (default: {-1})
        silent {bool} -- verbosity (default: {False})
    Raises:
        Exception: Unknown quality metric
    Returns:
        [type] -- points, normals, transforms, roll_angles, standoffs, collisions, quality
    """


    # sample points on the mesh and normals from them
    #points, face_indices = mesh.sample(number_of_grasps, return_index=True)
    points, face_indices = trimesh.sample.sample_surface_even(mesh, number_of_grasps)
    normals = mesh.face_normals[face_indices]

    number_of_grasps = len(points) # overwrite
    # sample angles
    alpha = np.random.uniform(low=alpha_lim[0], high=alpha_lim[1], size=number_of_grasps)
    beta = np.random.uniform(low=beta_lim[0], high=beta_lim[1], size=number_of_grasps)
    gamma = np.random.uniform(low=gamma_lim[0], high=gamma_lim[1], size=number_of_grasps)
    
    standoffs = np.random.uniform(low=gripper.standoff_range[0], high=gripper.standoff_range[1], size=number_of_grasps)

    origins = np.empty([number_of_grasps, 3])
    transforms = np.empty([number_of_grasps, 4, 4])
    quaternions = np.empty([number_of_grasps, 4])
    for k in range(number_of_grasps):
        origins[k] = points[k] + normals[k] * standoffs[k]
        angle_x = tra.quaternion_matrix(
             tra.quaternion_about_axis(alpha[k], [0, 0, 1]))
        angle_y = tra.quaternion_matrix(
             tra.quaternion_about_axis(beta[k], [0, 1, 0]))
        angle_z = tra.quaternion_matrix(
             tra.quaternion_about_axis(gamma[k], [1, 0, 0]))
        s2 = np.dot(tra.translation_matrix(origins[k]), trimesh.geometry.align_vectors([0, 0, -1], normals[k]) )
        _transform = np.dot( np.dot(np.dot( s2, angle_x), angle_y), angle_z)
        transforms[k] = _transform
        quaternions[k] = R.from_matrix(_transform[:3,:3]).as_quat()

    # check collisions between obj and gripper for given transforms
    collisions, _ = in_collision_with_gripper(mesh, transforms, gripper=gripper, silent=silent)

    qualities = grasp_quality_weighted_antipodal(
        transforms, collisions, object_mesh=mesh, gripper=gripper, silent=silent)

    qualities = np.array(qualities)
    collisions = np.array(collisions)

    return points, normals, transforms, origins, quaternions, alpha, beta, gamma, standoffs, qualities

def sample_multiple_grasps_bowl(mesh, gripper,
                            number_of_grasps=1000,
                            alpha_lim = [0, 2*np.pi],
                            beta_lim = [0, 0],
                            gamma_lim = [0, 0],
                            silent=False):
    """Sample a set of grasps for an object.
    Arguments:
        number_of_candidates {int} -- Number of grasps to sample
        mesh {trimesh} -- Object mesh
        gripper_name {str} -- Name of gripper model
        systematic_sampling {bool} -- Whether to use grid sampling for roll
    Keyword Arguments:
        surface_density {float} -- surface density, in m^2 (default: {0.005*0.005})
        standoff_density {float} -- density for standoff, in m (default: {0.01})
        roll_density {float} -- roll density, in deg (default: {15})
        quality_type {str} -- quality metric (default: {'antipodal'})
        min_quality {float} -- minimum grasp quality (default: {-1})
        silent {bool} -- verbosity (default: {False})
    Raises:
        Exception: Unknown quality metric
    Returns:
        [type] -- points, normals, transforms, roll_angles, standoffs, collisions, quality
    """
    # get the top edge of bowl
    vertices = np.array(mesh.vertices)
    coordinate = vertices[:,1]
    centroid = np.array([0,min(coordinate),0])
    upper_bound = max(coordinate) - 0.001
    edge_vertices = []
    orientations = []
    alpha = []
    toppest_point = [0,upper_bound,max(vertices[:,2])]

    for vertex in vertices:
        if vertex[1] >= upper_bound:
            edge_vertices.append(vertex)
            orientation = centroid - vertex
            orientation = orientation / np.linalg.norm(orientation)
            orientation = np.negative(orientation)
            #orientations.append(orientation)
            orientations.append([0,1,0])
            alpha.append(np.arctan2(vertex[2], vertex[0])+np.pi/2)
        if vertex[1] >= upper_bound/2 - 0.06 and vertex[1] <= upper_bound/2 + 0.06 and vertex[0] >= -0.01 and vertex[0] <= 0.01 and vertex[2] > 0:
            middle_point = vertex
    
    points = np.asarray(edge_vertices)
    normals = np.asarray(orientations)
    alpha = np.asarray(alpha)


    selected_indcs = np.random.choice(len(points), size=number_of_grasps)
    points = points[selected_indcs]
    normals = normals[selected_indcs]
    alpha = alpha[selected_indcs]

    beta = np.arctan2(toppest_point[2] - middle_point[2], toppest_point[1] - middle_point[1])

    number_of_grasps = len(points) # overwrite
    # sample angles
    #alpha = np.random.uniform(low=alpha_lim[0], high=alpha_lim[1], size=number_of_grasps)
    beta = np.random.uniform(low=beta-np.pi/20, high=beta+np.pi/20, size=number_of_grasps)
    gamma = np.random.uniform(low=gamma_lim[0], high=gamma_lim[1], size=number_of_grasps)
    
    #standoffs = np.random.uniform(low=gripper.standoff_range[0], high=gripper.standoff_range[1], size=number_of_grasps)
    standoffs = np.random.uniform(low=0.08, high=0.08, size=number_of_grasps)

    origins = np.empty([number_of_grasps, 3])
    transforms = np.empty([number_of_grasps, 4, 4])
    quaternions = np.empty([number_of_grasps, 4])
    for k in range(number_of_grasps):
        origins[k] = points[k]# + normals[k] * standoffs[k]
        angle_x = tra.quaternion_matrix(
             tra.quaternion_about_axis(alpha[k], [0, 0, 1]))
        angle_y = tra.quaternion_matrix(
             tra.quaternion_about_axis(beta[k], [0, 1, 0]))
        angle_z = tra.quaternion_matrix(
             tra.quaternion_about_axis(gamma[k], [1, 0, 0]))
        s2 = np.dot(tra.translation_matrix(origins[k]), trimesh.geometry.align_vectors([0, 0, -1], normals[k]) )
        _transform = np.dot( np.dot(np.dot( s2, angle_x), angle_y), angle_z)
   
        quaternions[k] = R.from_matrix(_transform[:3,:3]).as_quat()
        _r = R.from_matrix(_transform[:3,:3])
        translation_new = _r.apply([0,0,1])
        origins[k] = points[k] - standoffs[k] * translation_new   
        _transform[:3,3] = origins[k]
        transforms[k] = _transform
    # check collisions between obj and gripper for given transforms
    collisions, _ = in_collision_with_gripper(mesh, transforms, gripper=gripper, silent=silent)

    qualities = grasp_quality_weighted_antipodal(
        transforms, collisions, object_mesh=mesh, gripper=gripper, silent=silent)

    qualities = np.array(qualities)
    collisions = np.array(collisions)

    return points, normals, transforms, origins, quaternions, alpha, beta, gamma, standoffs, qualities

def sample_multiple_grasps_mug(mesh, gripper,
                            number_of_grasps=1000,
                            alpha_lim = [0, 2*np.pi],
                            beta_lim = [0, 0],
                            gamma_lim = [0, 0],
                            silent=False,
                            jcups=False):
    """Sample a set of grasps for an object.
    Arguments:
        number_of_candidates {int} -- Number of grasps to sample
        mesh {trimesh} -- Object mesh
        gripper_name {str} -- Name of gripper model
        systematic_sampling {bool} -- Whether to use grid sampling for roll
    Keyword Arguments:
        surface_density {float} -- surface density, in m^2 (default: {0.005*0.005})
        standoff_density {float} -- density for standoff, in m (default: {0.01})
        roll_density {float} -- roll density, in deg (default: {15})
        quality_type {str} -- quality metric (default: {'antipodal'})
        min_quality {float} -- minimum grasp quality (default: {-1})
        silent {bool} -- verbosity (default: {False})
    Raises:
        Exception: Unknown quality metric
    Returns:
        [type] -- points, normals, transforms, roll_angles, standoffs, collisions, quality
    """
    
    vertices = np.array(mesh.vertices)
    coordinate = vertices[:,1]
    upper_bound = max(coordinate) - 0.001
    top_edge_vertices = []
    top_edge_orientations = []
    handle_vertices = []
    handle_orientations_1 = []
    handle_orientations_2 = []
    handle_orientations_3 = []
    alpha = []
    centroid_middle = np.array([(np.median(vertices[:,0])),np.median(coordinate),(np.median(vertices[:,2]))])
    # print(f"center: {centroid}")
    lower_bound_z = min(vertices[:,2])

    for vertex in vertices:
        # get the top edge of mug
        if vertex[1] >= upper_bound:
            alpha.append(np.arctan2(vertex[2], vertex[0])+np.pi/2)
            top_edge_vertices.append(vertex)
            top_edge_orientations.append([0,1,0])

        # get the handle of mug
        if vertex[2] >= lower_bound_z and vertex[2] <= lower_bound_z/1.5:
            handle_vertices.append(vertex)
            orientation = vertex - centroid_middle
            orientation = orientation / np.linalg.norm(orientation)
            handle_orientations_1.append(orientation)
            handle_orientations_2.append([1,0,0])
            handle_orientations_3.append([-1,0,0])
    
    if not jcups:
        top_points, top_normals, top_transforms, top_origins, top_quaternions, top_alpha, top_beta, top_gamma, top_standoffs = \
            get_grasps_with_points_normals(gripper,top_edge_vertices,top_edge_orientations,number_of_grasps=number_of_grasps//4,alpha_lim=alpha_lim,beta_lim=beta_lim,gamma_lim=gamma_lim,alpha=alpha)

        handle_points_1, handle_normals_1, handle_transforms_1, handle_origins_1, handle_quaternions_1, handle_alpha_1, handle_beta_1, handle_gamma_1, handle_standoffs_1 = \
            get_grasps_with_points_normals(gripper,handle_vertices,handle_orientations_1,number_of_grasps=number_of_grasps//4,alpha_lim=[-np.pi/4,np.pi/4], beta_lim=beta_lim,gamma_lim=gamma_lim)

        handle_points_2, handle_normals_2, handle_transforms_2, handle_origins_2, handle_quaternions_2, handle_alpha_2, handle_beta_2, handle_gamma_2, handle_standoffs_2 = \
            get_grasps_with_points_normals(gripper,handle_vertices,handle_orientations_2,number_of_grasps=number_of_grasps//4,alpha_lim=[0,0], beta_lim=beta_lim,gamma_lim=gamma_lim,standoff=0.1)

        handle_points_3, handle_normals_3, handle_transforms_3, handle_origins_3, handle_quaternions_3, handle_alpha_3, handle_beta_3, handle_gamma_3, handle_standoffs_3 = \
            get_grasps_with_points_normals(gripper,handle_vertices,handle_orientations_3,number_of_grasps=number_of_grasps//4,alpha_lim=[0,0], beta_lim=beta_lim,gamma_lim=gamma_lim,standoff=0.1)

        points = np.concatenate([top_points,handle_points_1,handle_points_2,handle_points_3])
        normals = np.concatenate([top_normals,handle_normals_1,handle_normals_2,handle_normals_3])
        transforms = np.concatenate([top_transforms,handle_transforms_1,handle_transforms_2,handle_transforms_3])
        origins = np.concatenate([top_origins,handle_origins_1,handle_origins_2,handle_origins_3])
        quaternions = np.concatenate([top_quaternions,handle_quaternions_1,handle_quaternions_2,handle_quaternions_3])
        alpha = np.concatenate([top_alpha,handle_alpha_1,handle_alpha_2,handle_alpha_3])
        beta = np.concatenate([top_beta,handle_beta_1,handle_beta_2,handle_beta_3])
        gamma = np.concatenate([top_gamma,handle_gamma_1,handle_gamma_2,handle_gamma_3])
        standoffs = np.concatenate([top_standoffs,handle_standoffs_1,handle_standoffs_2,handle_standoffs_3])
    else:
        points, normals, transforms, origins, quaternions, alpha, beta, gamma, standoffs = \
            get_grasps_with_points_normals(gripper,top_edge_vertices,top_edge_orientations,number_of_grasps=number_of_grasps,alpha_lim=alpha_lim,beta_lim=beta_lim,gamma_lim=gamma_lim,alpha=alpha)

    # check collisions between obj and gripper for given transforms
    collisions, _ = in_collision_with_gripper(mesh, transforms, gripper=gripper, silent=silent)

    qualities = grasp_quality_weighted_antipodal(
        transforms, collisions, object_mesh=mesh, gripper=gripper, silent=silent)

    qualities = np.array(qualities)
    collisions = np.array(collisions)

    return points, normals, transforms, origins, quaternions, alpha, beta, gamma, standoffs, qualities

def get_grasps_with_points_normals(gripper,
                            points, normals,
                            number_of_grasps=1000,
                            alpha_lim = [0, 2*np.pi],
                            beta_lim = [0, 0],
                            gamma_lim = [0, 0],
                            standoff=0,alpha=None):
    points = np.array(points)
    points[:,2] -= np.random.uniform(low=0,high=standoff,size=len(points))
    normals = np.asarray(normals)

    selected_indcs = np.random.choice(len(points), size=number_of_grasps)
    number_of_grasps = len(selected_indcs)
    points = points[selected_indcs]
    normals = normals[selected_indcs]

    # sample angles
    if not isinstance(alpha,list): 
        alpha = np.random.uniform(low=alpha_lim[0], high=alpha_lim[1], size=number_of_grasps)
    else:
        alpha = np.asarray(alpha)
        alpha = alpha[selected_indcs]
        points[:,2] += np.random.uniform(low=0,high=0.01,size=len(points))
            
    beta = np.random.uniform(low=beta_lim[0], high=beta_lim[1], size=number_of_grasps)
    gamma = np.random.uniform(low=gamma_lim[0], high=gamma_lim[1], size=number_of_grasps)
    
    standoffs = np.random.uniform(low=gripper.standoff_range[0], high=gripper.standoff_range[1], size=number_of_grasps)

    origins = np.empty([number_of_grasps, 3])
    transforms = np.empty([number_of_grasps, 4, 4])
    quaternions = np.empty([number_of_grasps, 4])
    for k in range(number_of_grasps):
        origins[k] = points[k] + normals[k] * standoffs[k]
        angle_x = tra.quaternion_matrix(
            tra.quaternion_about_axis(alpha[k], [0, 0, 1]))
        angle_y = tra.quaternion_matrix(
            tra.quaternion_about_axis(beta[k], [0, 1, 0]))
        angle_z = tra.quaternion_matrix(
            tra.quaternion_about_axis(gamma[k], [1, 0, 0]))
        s2 = np.dot(tra.translation_matrix(origins[k]), trimesh.geometry.align_vectors([0, 0, -1], normals[k]) )
        _transform = np.dot( np.dot(np.dot( s2, angle_x), angle_y), angle_z)
        transforms[k] = _transform
        quaternions[k] = R.from_matrix(_transform[:3,:3]).as_quat()
    return points, normals, transforms, origins, quaternions, alpha, beta, gamma, standoffs

def perturb_grasp_locally(mesh, gripper,number_of_grasps, normals, origins, alpha, beta, gamma,
                        alpha_range=np.pi/36,
                        beta_range=np.pi/36,
                        gamma_range=np.pi/36,
                        t_range = 0.005,
                        silent=False):

    total_grasps = len(origins)*number_of_grasps
    new_origins = np.empty([total_grasps, 3])
    new_transforms = np.empty([total_grasps, 4, 4])
    new_quaternions = np.empty([total_grasps, 4])
    for i in range(len(origins)):
        # perturb eulers
        new_alpha = np.random.uniform(low=-alpha_range, high=alpha_range, size=number_of_grasps) + alpha[i]
        new_alpha[new_alpha < 0.0] = 0.0
        new_alpha[new_alpha > 2*np.pi] = 2*np.pi
        new_beta = np.random.uniform(low=-beta_range, high=beta_range, size=number_of_grasps) + beta[i]
        new_beta[new_beta < -np.pi] = -np.pi
        new_beta[new_beta > np.pi] = np.pi
        new_gamma = np.random.uniform(low=-gamma_range, high=gamma_range, size=number_of_grasps) + gamma[i]
        new_gamma[new_gamma < -np.pi] = -np.pi
        new_gamma[new_gamma > np.pi] = np.pi

        # perturb translations
        x = np.random.uniform(low=-t_range, high=t_range, size=number_of_grasps) + origins[i, 0]
        y = np.random.uniform(low=-t_range, high=t_range, size=number_of_grasps) + origins[i, 1]
        z = np.random.uniform(low=-t_range, high=t_range, size=number_of_grasps) + origins[i, 2]

        xyz = np.vstack([x,y,z])
    
        for k in range(number_of_grasps):
            new_origins[i*number_of_grasps + k] = xyz[:,k]
            angle_x = tra.quaternion_matrix(
                tra.quaternion_about_axis(new_alpha[k], [0, 0, 1]))
            angle_y = tra.quaternion_matrix(
                tra.quaternion_about_axis(new_beta[k], [0, 1, 0]))
            angle_z = tra.quaternion_matrix(
                tra.quaternion_about_axis(new_gamma[k], [1, 0, 0]))
            s2 = np.dot(tra.translation_matrix(new_origins[i*number_of_grasps + k]), trimesh.geometry.align_vectors([0, 0, -1], normals[i]) )
            _transform = np.dot( np.dot(np.dot( s2, angle_x), angle_y), angle_z)
            new_transforms[i*number_of_grasps + k] = _transform
            new_quaternions[i*number_of_grasps + k] = R.from_matrix(_transform[:3,:3]).as_quat()
        
    # check collisions between obj and gripper for given transforms
    collisions, _ = in_collision_with_gripper(mesh, new_transforms, gripper=gripper, silent=silent)

    qualities = grasp_quality_weighted_antipodal(
        new_transforms, collisions, object_mesh=mesh, gripper=gripper, silent=silent)

    qualities = np.array(qualities)
    #collisions = np.array(collisions)

    del new_alpha, new_beta, new_gamma, collisions
    return new_transforms, new_origins, new_quaternions, qualities


def perturb_transform(transform, number_of_grasps, 
                      min_translation=(-0.03,-0.03,-0.03),
                      max_translation=(0.03,0.03,0.03),
                      min_rotation=(-0.6,-0.2,-0.6),
                      max_rotation=(+0.6,+0.2,+0.6)):
    """
      Self explanatory.
    """
    perturbed_transforms = []
    for _ in range(number_of_grasps):
        sampled_translation = [np.random.uniform(lb, ub) for lb, ub in zip(min_translation, max_translation)]
        sampled_rotation = [np.random.uniform(lb, ub) for lb, ub in zip(min_rotation, max_rotation)]
        transform_new = tra.euler_matrix(*sampled_rotation)
        transform_new[:3, 3] = sampled_translation
        perturbed_transforms.append(np.matmul(transform, transform_new))
    
    return perturbed_transforms


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
    for tf in tqdm(gripper_transforms, disable=silent):
        min_distance.append(np.min([manager.min_distance_single(
            gripper_mesh, transform=tf) for gripper_mesh in gripper_meshes]))
    return [d == 0 for d in min_distance], min_distance


# sample random images
def uniform_a_b_sample(a,b, nsamples):
    return (b-a)*np.random.random(nsamples) + a

def gripper_handle(quality=None):
    if quality is not None:
        _B = quality*1.0
        _R = 1.0-_B
        _G = 0
        color = [_R, _G, _B, 1.0]
    else:
        color=None
        
    gripper_line_points_handle_part = np.array([
        [-0.0, 0.0000599553131906, 0.0672731392979622],
        [-0.0, 0.0000599553131906, -0.0032731392979622]
    ])
    small_gripper_handle_part = trimesh.path.Path3D(entities=[trimesh.path.entities.Line([0, 1], color=color)],
                                            vertices = gripper_line_points_handle_part)
    return small_gripper_handle_part


def gripper_bd(quality=None):

# def gripper_bd(quality=None, opacity=1.0):
    # do not change!
    # gripper_line_points_main_part = np.array([
    #     [0.0501874312758446, -0.0000599553131906, 0.1055731475353241],
    #     [0.0501874312758446, -0.0000599553131906, 0.0632731392979622],
    #     [-0.0501874312758446, 0.0000599553131906, 0.0632731392979622],
    #     [-0.0501874312758446, 0.0000599553131906, 0.0726731419563293],
    #     [-0.0501874312758446, 0.0000599553131906, 0.1055731475353241],
    # ])
    gripper_line_points_main_part = np.array([
        [0.0401874312758446, -0.0000599553131906, 0.1055731475353241],
        [0.0401874312758446, -0.0000599553131906, 0.0672731392979622],
        [-0.0401874312758446, 0.0000599553131906, 0.0672731392979622],
        [-0.0401874312758446, 0.0000599553131906, 0.0726731419563293],
        [-0.0401874312758446, 0.0000599553131906, 0.1055731475353241],
    ])


    gripper_line_points_handle_part = np.array([
        [-0.0, 0.0000599553131906, 0.0672731392979622],
        [-0.0, 0.0000599553131906, -0.0032731392979622]
    ])
    

    if quality is not None:
        _B = quality*1.0
        _R = 1.0-_B
        _G = 0
        color = [_R, _G, _B, opacity]
    else:
        color = None
    small_gripper_main_part = trimesh.path.Path3D(entities=[trimesh.path.entities.Line([0,1,2,3,4], color=color)],
                                                vertices = gripper_line_points_main_part)
    small_gripper_handle_part = trimesh.path.Path3D(entities=[trimesh.path.entities.Line([0, 1], color=color)],
                                                vertices = gripper_line_points_handle_part)
    small_gripper = trimesh.path.util.concatenate([small_gripper_main_part,
                                    small_gripper_handle_part])
    return small_gripper


def farthest_points(data,
                    nclusters,
                    dist_func,
                    return_center_indexes=False,
                    return_distances=False,
                    verbose=False):
    """
      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of 
          clusters.
        return_distances: bool, If True, return distances of each point from centers.
      
      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in 
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of 
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(data.shape[0],
                             dtype=np.int32), np.arange(data.shape[0],
                                                        dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0], ), dtype=np.int32) * -1
    distances = np.ones((data.shape[0], ), dtype=np.float32) * 1e7
    centers = []
    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter
        if verbose:
            print('farthest points max distance : {}'.format(
                np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters


def distance_by_translation_grasp(p1, p2):
    """
      Gets two nx4x4 numpy arrays and computes the translation of all the
      grasps.
    """
    t1 = p1[:, :3, 3]
    t2 = p2[:, :3, 3]
    return np.sqrt(np.sum(np.square(t1 - t2), axis=-1))


def distance_by_translation_point(p1, p2):
    """
      Gets two nx3 points and computes the distance between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))


def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates whether to use farthest point sampling
      to downsample the points. Farthest point sampling version runs slower.
    """
    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(pc,
                                                npoints,
                                                distance_by_translation_point,
                                                return_center_indexes=True)
        else:
            center_indexes = np.random.choice(range(pc.shape[0]),
                                              size=npoints,
                                              replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc

def sample_space_equally_for_grasp(extend = 0.35, step = 0.025, quats_per_point = 5, return_transforms = True):
    '''
    Generate equally spaced grasps between [-extend, extend] with step size by translation.
    Quaternions are randomly added to to each translation to build grasp.

    Inputs:
        - extends: float,
        - step: float
        - quats_per_point: int
        - return_transforms: boolean
    '''

    n_points = int((2*extend+step)/step)
    uniform_quaternions = genfromtxt('assets/data3_36864.qua', delimiter='\t')
    _x = np.linspace(-extend, extend, n_points)
    _y = np.linspace(-extend, extend, n_points)
    _z = np.linspace(-extend, extend, n_points)
    space_dimensions = [_x, _y, _z]

    translations = []
    quaternions = []
    for t in itertools.product(*space_dimensions):
        quat_indcs = np.random.choice(len(uniform_quaternions), size=quats_per_point)
        for quat_ind in quat_indcs:
            translations.append(t)
            quaternions.append( uniform_quaternions[quat_ind] )

    translations = np.array(translations)
    quaternions = np.array(quaternions)

    if return_transforms:
        # return transforms too
        transforms = np.repeat(np.eye(4)[:,:,np.newaxis], len(translations), axis=2).transpose([2,0,1])
        transforms[:,:3,3] = translations
        rot = R.from_quat(quaternions).as_matrix()
        transforms[:,:3,:3] = rot

        return quaternions, translations, transforms

    return quaternions, translations

def sample_equal_spaced_grasps(mesh, gripper,
                              extend = 0.35,
                              step = 0.025,
                              quats_per_point = 5,
                              silent=False):
    """
    Sample a set of grasps for an object in equal spaced 
    """
    quaternions, translations, transforms = sample_space_equally_for_grasp(extend=extend, step = step, quats_per_point = quats_per_point)

    # check collisions between obj and gripper for given transforms
    collisions, _ = in_collision_with_gripper(mesh, transforms, gripper=gripper, silent=silent)

    qualities = grasp_quality_weighted_antipodal(
        transforms, collisions, object_mesh=mesh, gripper=gripper, silent=silent)

    qualities = np.array(qualities)
    collisions = np.array(collisions)

    return quaternions, translations, transforms, qualities

import json

def get_scale(path_to_json):
    with open(path_to_json) as json_file:
        info = json.load(json_file)
        return  info["scale"]