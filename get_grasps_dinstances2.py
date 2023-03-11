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

def get_transform(translation,orientation):
	transform = np.eye(4)
	r = R.from_quat(orientation)
	transform[:3,:3] = r.as_matrix()
	transform[:3,3] = translation
	return transform

def get_transforms(data):
	translations = data["translations"]
	orientations = data["quaternions"]
	transforms = []
	for t,q in zip(translations,orientations):
		transform = get_transform(t,q)
		transforms.append(transform)
	return transforms

def get_distances(data):
	transforms = get_transforms(data)
	distances = np.asarray(sampler.get_minimum_distance_to_obj(transforms))
	return np.asarray(distances)

object = "sugar_box"

object_to_code = {
	"sugar_box"      :"004_sugar_box",
	"bowl"           :"024_bowl",
	"tomato_soup_can":'005_tomato_soup_can', 
	"potted_meat_can":'010_potted_meat_can', 
	"mug"            :'025_mug', 
	"foam_brick"     : '061_foam_brick', 
	"j_cups"         :'065_j_cups',
	"sponge"         :'026_sponge'
	}


obj_filename = f"assets/meshes_10_objects/{object_to_code[object]}/google_16k/textured.obj"
info, args = pickle.load(open('../isaac/experiments/results/experiment1/004_sugar_box/1641738054/info', 'rb'))
# grasps_distances = np.asarray(pickle.load(open('gripper_to_obj_distances.pkl', 'rb')))
print(info.keys())

success_last = info['success'][-1, :]
mask = np.argsort(success_last)[::-1][:50]

translations_final = info['translations'][-1, mask, :]
quaternions_final = info['quaternions'][-1, mask, :]
# for key in graspnet_grasps_initial_data:
# 	print(key)
# print(graspnet_grasps_initial_data.keys())


# success_last = info['success'][-1, :]
# mask = np.argsort(success_last)[::-1][:50]
# translations_final = info['translations'][-1, mask, :]
# quaternions_final = info['quaternions'][-1, mask, :]

transforms = []
for t,q in zip(translations_final,quaternions_final):
	transform = get_transform(t,q)
	transforms.append(transform)


sampler = GraspSampler()
# load object
sampler.update_object(obj_filename=obj_filename, name=obj_filename, obj_scale=1)

distances = sampler.get_minimum_distance_to_obj(transforms)

with open("../isaac/gripper_to_obj_distances.pkl","wb") as handle:
	pickle.dump(distances,handle)
print(distances.shape)
