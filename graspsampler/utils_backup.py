from typing import final
import numpy as np
from numpy.random.mtrand import normal
import trimesh.transformations as tra
from tqdm import tqdm
import trimesh
from scipy.spatial.transform import Rotation as R

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

def grasp_quality_antipodal_single_contact(transforms, collisions, object_mesh, gripper, silent=False):
    """Grasp quality function.
    Arguments:
        transforms {numpy.array} -- grasps
        collisions {list of bool} -- collision information
        object_mesh {trimesh} -- object mesh
    Keyword Arguments:
        gripper_name {str} -- name of gripper (default: {'panda'})
        silent {bool} -- verbosity (default: {False})
    Returns:
        list of float -- quality of grasps [0..1]
    """
    res = []
    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
            object_mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)
    locations = []
    for p, colliding in tqdm(zip(transforms, collisions), total=len(transforms), disable=silent):
        if colliding:
            res.append(0)
        else:
            ray_origins, ray_directions = gripper.get_closing_rays(p)
            locations, index_ray, index_tri = intersector.intersects_location(
                ray_origins, ray_directions, multiple_hits=False)

            if locations.size == 0:
                res.append(0)
            else:
                # chose contact points for each finger [they are stored in an alternating fashion]
                index_ray_left = np.array([i for i, num in enumerate(
                    index_ray) if num % 2 == 0 and np.linalg.norm(ray_origins[num]-locations[i]) < 2.0*gripper.q])
                index_ray_right = np.array([i for i, num in enumerate(
                    index_ray) if num % 2 == 1 and np.linalg.norm(ray_origins[num]-locations[i]) < 2.0*gripper.q])

                if index_ray_left.size == 0 or index_ray_right.size == 0:
                    res.append(0)
                else:
                    # select the contact point closest to the finger (which would be hit first during closing)
                    left_contact_idx = np.linalg.norm(
                        ray_origins[index_ray[index_ray_left]] - locations[index_ray_left], axis=1).argmin()
                    right_contact_idx = np.linalg.norm(
                        ray_origins[index_ray[index_ray_right]] - locations[index_ray_right], axis=1).argmin()
                    left_contact_point = locations[index_ray_left[left_contact_idx]]
                    right_contact_point = locations[index_ray_right[right_contact_idx]]

                    left_contact_normal = object_mesh.face_normals[index_tri[index_ray_left[left_contact_idx]]]
                    right_contact_normal = object_mesh.face_normals[
                        index_tri[index_ray_right[right_contact_idx]]]

                    l_to_r = (right_contact_point - left_contact_point) / \
                        np.linalg.norm(right_contact_point -
                                       left_contact_point)
                    r_to_l = (left_contact_point - right_contact_point) / \
                        np.linalg.norm(left_contact_point -
                                       right_contact_point)

                    qual_left = np.dot(left_contact_normal, r_to_l)
                    qual_right = np.dot(right_contact_normal, l_to_r)
                    # print(f"qual_left: {qual_left}")
                    # print(f"qual_right: {qual_right}")
                    if qual_left < 0 or qual_right < 0:
                        qual = 0
                    else:
                        # qual = qual_left * qual_right
                        qual = min(qual_left, qual_right)
                    # math.cos(math.atan(friction_coefficient))

                    res.append(qual)
    return res


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
                    contact_normals = object_mesh.face_normals[index_tri[valid_locations]]
                    motion_normals = ray_directions[index_ray[valid_locations]]
                    dot_prods = -(motion_normals * contact_normals).sum(axis=1)

                    #print(dot_prods)


                    # print(f"motion_normals: {motion_normals}")
                    # print(f"motion_normals.shape: {motion_normals.shape}")
                    # print(f"contact_normals: {contact_normals}")
                    # print(f"contact_normals.shape: {contact_normals.shape}")
                    # print(f"dot_prods: {dot_prods}")
                    # print(f"dot_prods.shape: {dot_prods.shape}")
                    # print(f"np.cos(dot_prods): {np.cos(dot_prods)}")
                    res.append(dot_prods.sum() / len(ray_origins))
                    # res.append(dot_prods.sum() / len(ray_origins))
    return res


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
                    contact_normals = object_mesh.face_normals[index_tri[valid_locations]]
                    motion_normals = ray_directions[index_ray[valid_locations]]
                    dot_prods = -(motion_normals * contact_normals).sum(axis=1)

                    #print(dot_prods)


                    # print(f"motion_normals: {motion_normals}")
                    # print(f"motion_normals.shape: {motion_normals.shape}")
                    # print(f"contact_normals: {contact_normals}")
                    # print(f"contact_normals.shape: {contact_normals.shape}")
                    # print(f"dot_prods: {dot_prods}")
                    # print(f"dot_prods.shape: {dot_prods.shape}")
                    # print(f"np.cos(dot_prods): {np.cos(dot_prods)}")
                    res.append(dot_prods.sum() / len(ray_origins))
                    # res.append(dot_prods.sum() / len(ray_origins))
    return res


def sample_multiple_grasps_light(obj, gripper,
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

    # prepare gripper utilities
    gripper_big = gripper.get_bb(all=True)
    gripper_small = gripper.get_bb(all=False)

    # sample points on the mesh and normals from them
    points, face_indices = obj.mesh.sample(number_of_grasps, return_index=True)
    normals = obj.mesh.face_normals[face_indices]

    
    # sample angles
    alpha = np.random.uniform(low=alpha_lim[0], high=alpha_lim[1], size=number_of_grasps)
    beta = np.random.uniform(low=beta_lim[0], high=beta_lim[1], size=number_of_grasps)
    gamma = np.random.uniform(low=gamma_lim[0], high=gamma_lim[1], size=number_of_grasps)
    standoffs = np.random.uniform(low=gripper.standoff_range[0], high=gripper.standoff_range[1], size=number_of_grasps)

    origins = np.empty([number_of_grasps, 3])
    transforms = np.empty([number_of_grasps, 4, 4])
    quaternions = np.empty([number_of_grasps, 4])
    is_promising = np.empty([number_of_grasps])
    for k in range(number_of_grasps):
        origins[k] = points[k] + normals[k] * standoffs[k]
        angle_x = tra.quaternion_matrix(
             tra.quaternion_about_axis(alpha[k], [0, 0, 1]))
        s2 = np.dot(tra.translation_matrix(origins[k]), trimesh.geometry.align_vectors([0, 0, -1], normals[k]) )
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

        # check collisions
        promising_label = 1
        if not obj.in_collision_with(gripper_big, transform=transforms[k]):
            # away from gripper
            promising_label = 0
            # collision with gripper
        if obj.in_collision_with(gripper_small, transform=transforms[k]):
            promising_label = -1

        # obj.min_distance_single(gripper_small, transform=transforms[k])

        is_promising[k] = promising_label


    return points, normals, transforms, origins, quaternions, alpha, standoffs, is_promising

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
    points, face_indices = mesh.sample(number_of_grasps, return_index=True)
    normals = mesh.face_normals[face_indices]

    
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


    qualities_1 = grasp_quality_weighted_antipodal(
        transforms, collisions, object_mesh=mesh, gripper=gripper, silent=silent)
    qualities_2 = grasp_quality_antipodal_single_contact(
        transforms, collisions, object_mesh=mesh, gripper=gripper, silent=silent)

    qualities_1 = np.array(qualities_1)
    qualities_2 = np.array(qualities_2)
    collisions = np.array(collisions)

    return points, normals, transforms, origins, quaternions, alpha, standoffs, collisions, qualities_1, qualities_2

def perturb_grasp_locally(mesh, gripper,number_of_grasps, normals, origins, alpha,
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
        beta = np.random.uniform(low=-beta_range, high=beta_range, size=number_of_grasps)
        beta[beta < -np.pi] = -np.pi
        beta[beta > np.pi] = np.pi
        gamma = np.random.uniform(low=-gamma_range, high=gamma_range, size=number_of_grasps)
        gamma[gamma < -np.pi] = -np.pi
        gamma[gamma > np.pi] = np.pi

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
                tra.quaternion_about_axis(beta[k], [0, 1, 0]))
            angle_z = tra.quaternion_matrix(
                tra.quaternion_about_axis(gamma[k], [1, 0, 0]))
            s2 = np.dot(tra.translation_matrix(new_origins[i*number_of_grasps + k]), trimesh.geometry.align_vectors([0, 0, -1], normals[i]) )
            _transform = np.dot( np.dot(np.dot( s2, angle_x), angle_y), angle_z)
            new_transforms[i*number_of_grasps + k] = _transform
            new_quaternions[i*number_of_grasps + k] = R.from_matrix(_transform[:3,:3]).as_quat()
        
    # check collisions between obj and gripper for given transforms
    collisions, _ = in_collision_with_gripper(mesh, new_transforms, gripper=gripper, silent=silent)

    qualities_1 = grasp_quality_weighted_antipodal(
        new_transforms, collisions, object_mesh=mesh, gripper=gripper, silent=silent)
    qualities_2 = grasp_quality_antipodal_single_contact(
        new_transforms, collisions, object_mesh=mesh, gripper=gripper, silent=silent)

    qualities_1 = np.array(qualities_1)
    qualities_2 = np.array(qualities_2)
    collisions = np.array(collisions)
    return new_transforms, new_origins, new_quaternions, collisions, qualities_1, qualities_2

def perturb_grasp(mesh, gripper, normal, point,
                  standoff, standoff_range=0.1,
                  angle=None, angle_range=0.1,
                  quality_type='weighted_antipodal', number_of_grasps=100, silent=False):
    random_numbers = np.random.random(number_of_grasps)
    points = []
    normals = []
    transforms = []
    #quaternions = []
    #origins = []
    roll_angles = []
    standoffs = []
    for random_number in random_numbers:
        points.append(point)
        normals.append(normal)
        print(standoff, 2*standoff_range*random_number - standoff_range + standoff)
        # perturb orientation
        roll_angles.append(2*angle_range*random_number - angle_range+angle)
        orientation = tra.quaternion_matrix(tra.quaternion_about_axis(roll_angles[-1], [0, 0, 1]))
        

        # perturb standoff
        standoffs.append(2*standoff_range*random_number - standoff_range + standoff)
        origin = point+normal*standoffs[-1]

        # get new transform
        transforms.append( np.dot(np.dot(tra.translation_matrix(origin),
                           trimesh.geometry.align_vectors([0, 0, -1], normal)),
                           orientation) )
        # this one leads to Gimbal Lock https://en.wikipedia.org/wiki/Gimbal_lock
        
    # check collisions between obj and gripper for given transforms
    collisions, _ = in_collision_with_gripper(mesh, transforms, gripper=gripper, silent=silent)    
    

    qualities_1 = grasp_quality_weighted_antipodal(
        transforms, collisions, object_mesh=mesh, gripper=gripper, silent=silent)
    qualities_2 = grasp_quality_antipodal_single_contact(
        transforms, collisions, object_mesh=mesh, gripper=gripper, silent=silent)

    points = np.array(points)
    normals = np.array(normals)
    transforms = np.array(transforms)
    roll_angles = np.array(roll_angles)
    standoffs = np.array(standoffs)
    collisions = np.array(collisions)
    #qualities = np.array(qualities)
    quality_types = [quality_type]*len(qualities_1)

    return points, normals, transforms, roll_angles, standoffs, collisions, qualities_1, qualities_2, quality_types
    # return points, normals, origins, quaternions, roll_angles, standoffs, collisions, qualities_1, qualities_2, quality_types







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


def gripper_bd(quality=None):

    # do not change!
    gripper_line_points_main_part = np.array([
        [0.0501874312758446, -0.0000599553131906, 0.1055731475353241],
        [0.0501874312758446, -0.0000599553131906, 0.0632731392979622],
        [-0.0501874312758446, 0.0000599553131906, 0.0632731392979622],
        [-0.0501874312758446, 0.0000599553131906, 0.0726731419563293],
        [-0.0501874312758446, 0.0000599553131906, 0.1055731475353241],
    ])

    gripper_line_points_handle_part = np.array([
        [-0.0, 0.0000599553131906, 0.0632731392979622],
        [-0.0, 0.0000599553131906, -0.0032731392979622]
    ])

    if quality is not None:
        _B = quality*1.0
        _R = 1.0-_B
        _G = 0
        color = [_R, _G, _B, 1.0]
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
    points, face_indices = mesh.sample(number_of_grasps, return_index=True)
    normals = mesh.face_normals[face_indices]

    
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


    qualities_1 = grasp_quality_weighted_antipodal(
        transforms, collisions, object_mesh=mesh, gripper=gripper, silent=silent)
    qualities_2 = grasp_quality_antipodal_single_contact(
        transforms, collisions, object_mesh=mesh, gripper=gripper, silent=silent)

    qualities_1 = np.array(qualities_1)
    qualities_2 = np.array(qualities_2)
    collisions = np.array(collisions)
    



    return points, normals, transforms, origins, quaternions, alpha, standoffs, collisions, qualities_1, qualities_2


def purturb_single_grasp_locally_test(mesh, gripper,number_of_grasps, normal, origin, alpha,
                        alpha_range=np.pi/36,
                        beta_range=np.pi/36,
                        gamma_range=np.pi/36,
                        t_range = 0.005,
                        silent=True):

    new_origins = np.empty([number_of_grasps, 3])
    new_transforms = np.empty([number_of_grasps, 4, 4])
    new_quaternions = np.empty([number_of_grasps, 4])


    new_alpha = np.random.uniform(low=-alpha_range, high=alpha_range, size=number_of_grasps) + alpha
    new_alpha[new_alpha < 0.0] = 0.0
    new_alpha[new_alpha > 2*np.pi] = 2*np.pi
    beta = np.random.uniform(low=-beta_range, high=beta_range, size=number_of_grasps)
    beta[beta < -np.pi] = -np.pi
    beta[beta > np.pi] = np.pi
    gamma = np.random.uniform(low=-gamma_range, high=gamma_range, size=number_of_grasps)
    gamma[gamma < -np.pi] = -np.pi
    gamma[gamma > np.pi] = np.pi

    # perturb translations
    x = np.random.uniform(low=-t_range, high=t_range, size=number_of_grasps) + origin[0]
    y = np.random.uniform(low=-t_range, high=t_range, size=number_of_grasps) + origin[1]
    z = np.random.uniform(low=-t_range, high=t_range, size=number_of_grasps) + origin[2]

    xyz = np.vstack([x,y,z])


def sample_multiple_grasps_test(mesh, gripper,
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
    points, face_indices = mesh.sample(number_of_grasps, return_index=True)
    # points, face_indices = trimesh.sample.sample_surface_even(mesh, number_of_grasps)
    normals = mesh.face_normals[face_indices]
    number_of_grasps = len(face_indices)

    
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


    qualities_1 = grasp_quality_weighted_antipodal(
        transforms, collisions, object_mesh=mesh, gripper=gripper, silent=silent)
    qualities_2 = grasp_quality_antipodal_single_contact(
        transforms, collisions, object_mesh=mesh, gripper=gripper, silent=silent)

    qualities_1 = np.array(qualities_1)
    qualities_2 = np.array(qualities_2)
    collisions = np.array(collisions)
    



    return points, normals, transforms, origins, quaternions, alpha, beta, gamma, standoffs, collisions, qualities_1, qualities_2

def perturb_grasp_locally_test(mesh, 
                        gripper, 
                        number_of_grasps, 
                        normals, 
                        qualities_1,
                        origins, 
                        alpha, 
                        beta, 
                        gamma,
                        recur_search_depth,
                        alpha_range=np.pi/36,
                        beta_range=np.pi/36,
                        gamma_range=np.pi/36,
                        t_range = 0.005,
                        silent=False):


    if recur_search_depth == 0 or origins == np.array([]):
        return np.array([]),np.array([]),np.array([]),np.array([]),np.array([])

    new_total_grasps = len(origins)*number_of_grasps
    print("new_total:", new_total_grasps)
    new_origins = np.empty([new_total_grasps, 3])
    new_transforms = np.empty([new_total_grasps, 4, 4])
    new_quaternions = np.empty([new_total_grasps, 4])
    new_alpha = np.empty([new_total_grasps])
    new_beta = np.empty([new_total_grasps])
    new_gamma = np.empty([new_total_grasps])
    for i in range(len(origins)):
        # perturb eulers
        new_alpha[i*number_of_grasps:i*number_of_grasps+number_of_grasps] = np.random.uniform(low=-alpha_range, high=alpha_range, size=number_of_grasps) + alpha[i]
        new_alpha[new_alpha < 0.0] = 0.0
        new_alpha[new_alpha > 2*np.pi] = 2*np.pi
        new_beta[i*number_of_grasps:i*number_of_grasps+number_of_grasps] = np.random.uniform(low=-beta_range, high=beta_range, size=number_of_grasps) + beta[i]
        new_beta[new_beta < -np.pi] = -np.pi
        new_beta[new_beta > np.pi] = np.pi
        new_gamma[i*number_of_grasps:i*number_of_grasps+number_of_grasps] = np.random.uniform(low=-gamma_range, high=gamma_range, size=number_of_grasps) + gamma[i]
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

    cur_qualities_1 = grasp_quality_weighted_antipodal(
        new_transforms, collisions, object_mesh=mesh, gripper=gripper, silent=silent)
    cur_qualities_2 = grasp_quality_antipodal_single_contact(
        new_transforms, collisions, object_mesh=mesh, gripper=gripper, silent=silent)

    cur_qualities_1 = np.array(cur_qualities_1)
    cur_qualities_2 = np.array(cur_qualities_2)
    collisions = np.array(collisions)

    better_grasps_indecies = np.array([])
    print("original grasps: ", len(qualities_1))
    print("current grasps: ", len(cur_qualities_1))
    print("current transforms: ", len(new_transforms))

    for origin_grasp_index in range(len(qualities_1)):
        # corresponding_new_grasps_indecies = [origin_grasp_index*number_of_grasps,origin_grasp_index*number_of_grasps+number_of_grasps]
        cur_better_grasps_index = []
        for i in range(origin_grasp_index*number_of_grasps,origin_grasp_index*number_of_grasps+number_of_grasps):
            if cur_qualities_1[i] > qualities_1[origin_grasp_index]:
                cur_better_grasps_index.append(i)
        cur_better_grasps_index = np.array(cur_better_grasps_index)
        better_grasps_indecies = np.concatenate([better_grasps_indecies, cur_better_grasps_index])
        # cur_better_grasps_indecies = np.where(cur_qualities_1[corresponding_new_grasps_indecies] > qualities_1[origin_grasp_index])[0] + corresponding_new_grasps_indecies[0]
        # better_grasps_indecies = np.concatenate([better_grasps_indecies, cur_better_grasps_indecies])
    better_grasps_indecies = better_grasps_indecies.astype(int)
    if better_grasps_indecies != np.array([]):  
        _,_, new_face_indecies = trimesh.proximity.closest_point(mesh, new_origins[better_grasps_indecies])


        better_grasps_qualities_1, better_grasps_qualities_2,better_grasps_origins,better_grasps_quaterions,  better_grasps_transforms= perturb_grasp_locally_test(
                                                                                                        mesh=mesh,
                                                                                                        gripper=gripper,
                                                                                                        number_of_grasps=number_of_grasps,
                                                                                                        qualities_1=cur_qualities_1[better_grasps_indecies],
                                                                                                        origins=new_origins[better_grasps_indecies],
                                                                                                        normals=mesh.face_normals[new_face_indecies],
                                                                                                        alpha=new_alpha[better_grasps_indecies],
                                                                                                        beta=new_beta[better_grasps_indecies],
                                                                                                        gamma=new_gamma[better_grasps_indecies],
                                                                                                        recur_search_depth=recur_search_depth-1,
                                                                                                        )
        if better_grasps_origins != np.array([]):
            cur_qualities_1 = np.concatenate([cur_qualities_1, better_grasps_qualities_1])
            cur_qualities_2 = np.concatenate([cur_qualities_2, better_grasps_qualities_2])
            print("better_grasps_origins shape: ", better_grasps_origins.shape)
            new_origins = np.concatenate([new_origins,better_grasps_origins])
            new_quaternions = np.concatenate([new_quaternions,better_grasps_quaterions])
            new_transforms = np.concatenate([new_transforms,better_grasps_transforms])

    return cur_qualities_1, cur_qualities_2, new_origins, new_quaternions, new_transforms


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

    qualities_1 = grasp_quality_weighted_antipodal(
        new_transforms, collisions, object_mesh=mesh, gripper=gripper, silent=silent)
    qualities_2 = grasp_quality_antipodal_single_contact(
        new_transforms, collisions, object_mesh=mesh, gripper=gripper, silent=silent)

    qualities_1 = np.array(qualities_1)
    qualities_2 = np.array(qualities_2)
    collisions = np.array(collisions)
    return new_transforms, new_origins, new_quaternions, collisions, qualities_1, qualities_2
