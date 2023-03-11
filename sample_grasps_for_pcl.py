import numpy as np
from trimesh.points import PointCloud
from math import degrees
import numpy as np
import trimesh.transformations as tra
from scipy.spatial.transform import Rotation as R
import trimesh


def sample_grasps_pcl(pcl,
					number_of_grasps=1000,
					alpha_lim = [0, np.pi],
					beta_lim = [0, 0],
					gamma_lim = [0, 0],
					silent=False):
    
	# get convex hull
	pcl = PointCloud(pcl)
	mesh = pcl.convex_hull

	# sample points on the mesh and normals from them
	points, face_indices = trimesh.sample.sample_surface_even(mesh, number_of_grasps)
	normals = mesh.face_normals[face_indices]

	number_of_grasps = len(points) # overwrite
	
	# sample angles
	alpha = np.random.uniform(low=alpha_lim[0], high=alpha_lim[1], size=number_of_grasps)
	beta = np.random.uniform(low=beta_lim[0], high=beta_lim[1], size=number_of_grasps)
	gamma = np.random.uniform(low=gamma_lim[0], high=gamma_lim[1], size=number_of_grasps)

	standoffs = np.random.uniform(low=0.05, high=0.1, size=number_of_grasps)

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

	return origins.flatten(), quaternions.flatten()



def sample_grasps_for_pcls(pcls, nun_of_grasps=1):
	translations = []
	quaternions = []

	for pcl in pcls:
		temp_translations, temp_quaternions = sample_grasps_pcl(pcl,number_of_grasps=nun_of_grasps,
																	alpha_lim = [0, np.pi],
																	beta_lim = [0, 0],
																	gamma_lim = [0, 0],
																	silent=False)
		translations.append(temp_translations)
		quaternions.append(temp_quaternions)
	
	return  np.array(quaternions), np.array(translations)

