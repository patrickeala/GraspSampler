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

cat = "mug"
idx = 12
obj_filename = f"grasp_data/meshes/{cat}/{cat}{idx:03}.obj"
metadata_filename = f'grasp_data/info/{cat}/{cat}{idx:03}.json'
metadata = json.load(open(metadata_filename,'r'))
save_dir = f"hardcoded_data/{cat}/{cat}{idx:03}"
positive_grasps = []

if not os.path.exists(save_dir):
	os.makedirs(save_dir)

def rotate_around_y_axis(translation,orientation,degree):
	_r = R.from_rotvec(degree* np.array([0, 1, 0]), degrees=True)
	orientation[1] += degree
	# _q = _r.as_quat()
	# r = R.from_euler("xyz",orientation,degrees=True)
	# ori_rot = r.as_matrix()

	# temp_rot = _r.as_matrix()
	# print("temp_rot: ",temp_rot)
	# new_rot = ori_rot*temp_rot
	# print("new_rot: ",new_rot)
	# euler_new = R.from_matrix(temp_rot).as_euler("xyz",degrees=True)
	# print("euler_new: ",euler_new)
	# q = r.as_quat()
	# rot_new =  _r.apply(r.as_matrix())
	# euler_new = R.from_matrix(rot_new).as_euler("xyz",degrees=True)
	translation_new = _r.apply(translation)
	# print(translation_new, euler_new, get_transform(translation_new,euler_new))
	return translation_new, orientation, get_transform(translation_new,orientation)

def get_transform(translation,orientation):
	transform = np.eye(4)
	r = R.from_euler("XYZ",orientation,degrees=True)
	transform[:3,:3] = r.as_matrix()
	transform[:3,3] = translation
	return transform

init_translation = np.array([0.0,0.0,0.0])
init_orientation = np.array([0.0,0.0,0.0])
transform = get_transform(init_translation,init_orientation)

sampler = GraspSampler()
# load object
sampler.update_object(obj_filename=obj_filename, name=obj_filename, obj_scale=metadata['scale'])

change = False

def listener():
	global init_translation
	global init_orientation
	global transform
	global change
	while 1:
		motion = input("put the motion of gripper here")
		if motion == "q":
			init_translation[0] += 0.01
		elif motion == "a":
			init_translation[0] -= 0.01								
		elif motion == "w":
			init_translation[1] += 0.01		
		elif motion == "s":
			init_translation[1] -= 0.01		
		elif motion == "e":
			init_translation[2] += 0.01	
		elif motion == "d":
			init_translation[2] -= 0.01	
		elif motion == "r":
			init_orientation[0] += 5
		elif motion == "f":
			init_orientation[0] -= 5
		elif motion == "t":
			init_orientation[1] += 5
		elif motion == "g":
			init_orientation[1] -= 5
		elif motion == "y":
			init_orientation[2] += 5
		elif motion == "h":
			init_orientation[2] -= 5
		print(init_orientation)
		transform = get_transform(init_translation,init_orientation)
		if motion == "u":
			init_translation, init_orientation, transform = rotate_around_y_axis(init_translation,init_orientation,degree=45)
			print("here")
		if motion == "m":
			cur_transform = np.array(transform)
			positive_grasps.append(cur_transform)
			print(positive_grasps)
			print(f"have got {len(positive_grasps)} positive grasps")
		elif motion == "done":
			np.savez_compressed(f'{save_dir}/hardcoded_grasps.npz',
					transforms=np.array(positive_grasps))
			print("done with storing grasps")
		print("done with updating position")

		print(transform)
		change = True
thread = threading.Thread(target=listener)
thread.start()



sampler.grasp_visualize(transform, coordinate_frame=True,grasp_debug_items=True,other_debug_items=False)
while 1:
	if change:
		change = False
		sampler.grasp_visualize(transform, coordinate_frame=True,grasp_debug_items=True,other_debug_items=False)
		obj_mesh = sampler.obj.mesh
		point = [transform[0,3],transform[1,3],transform[2,3]]
		data = trimesh.proximity.closest_point(obj_mesh,[point])
		point = data[0]
		# point_trimesh = trimesh.Trimesh(vertices=point)
		pc = trimesh.points.PointCloud(vertices=point,colors=[123,123,255,255])
		print(sampler.get_minimum_distance_to_obj([transform]))
		sampler.scene.add_rays(sampler.gripper.get_closing_rays())
		sampler.scene.add_geometry(pc)
		sampler.scene.show()
		print("point to query", point)
		distance = data[1]
		print("-------------------distance is :",distance[0])


# sampler.grasp_visualize(data["transforms"][index_to_check], coordinate_frame=True,grasp_debug_items=True,other_debug_items=False)






