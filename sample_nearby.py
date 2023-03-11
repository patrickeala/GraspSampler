import pickle as pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
from graspsampler.GraspSampler import GraspSampler
from graspsampler.common import Scene, Object, PandaGripper
import json
from pathlib import Path
import graspsampler.utils as utils
import time
from tqdm.auto import tqdm

# cat setup
category = "pan" # cylinder bowl mug bottle
number_of_positive_grasps = 1000
number_of_nearby_grasps = 100
num_objects = 21 if category == "mug" else 20


def qt2mat(q,t):
	transform = np.eye(4)
	rot_mat = R.from_quat(q).as_matrix()
	transform[:3,:3] = rot_mat
	transform[:3,3] = t
	return transform

def mat2qt(transform):
    rot_mat = transform[:3,:3]
    q = R.from_matrix(rot_mat).as_quat()
    t = transform[:3,3] 
    return q,t

def process(i):
	# set data path
	data_path = f"../isaac/grasp_data_generated/{category}/{category}{i:03}_isaac"

	main1_data_rest = f"{data_path}/main1_rest.npz"
	main2_data_rest = f"{data_path}/main2_rest.npz"
	main1_data = f"{data_path}/main1.npz"
	main2_data = f"{data_path}/main1.npz"
	main3_data_rest = f"{data_path}/main3_rest.npz"
	main4_data_rest = f"{data_path}/main4_rest.npz"
	main3_data = f"{data_path}/main3.npz"
	main4_data = f"{data_path}/main4.npz"
	even_data = f"{data_path}/main1_even_grasps.npz"
	# data_filenames = [main1_data_rest,main2_data_rest,main1_data,main2_data]
	data_filenames = [main1_data_rest,main2_data_rest,main2_data,main3_data,main4_data,main3_data_rest,main4_data_rest]

	# load data
	# data = np.load(even_data) 
	data = np.load(main1_data) 

	cur_quaternions = data["quaternions"]
	cur_translations = data["translations"]
	cur_isaac_labels = data["isaac_labels"]
	quaternions = cur_quaternions[cur_isaac_labels==1]
	translations = cur_translations[cur_isaac_labels==1]
	for data_filename in data_filenames:
		try:
			data = np.load(data_filename)
		except:
			continue
		cur_quaternions = data["quaternions"]
		cur_translations = data["translations"]
		cur_isaac_labels = data["isaac_labels"]
		positive_quaternions = cur_quaternions[cur_isaac_labels==1]
		positive_translations = cur_translations[cur_isaac_labels==1]
		print(positive_quaternions.shape)
		quaternions = np.concatenate([quaternions,positive_quaternions], axis=0)
		translations =  np.concatenate([translations,positive_translations],axis=0)

	if len(quaternions) > number_of_positive_grasps:
		_mask_positive = np.random.choice(len(quaternions), number_of_positive_grasps, replace=False)
		# print(len(quaternions))
		quaternions = quaternions[_mask_positive]
		translations = translations[_mask_positive]

	save_dir = f'../isaac/grasps_sampled_for_positive_grasps/{category}/{category}{i:03}'
	Path(save_dir).mkdir(parents=True, exist_ok=True)
	np.savez_compressed(f"{save_dir}/main_grasps",  quaternions=quaternions, translations=translations)


	obj_filename = f'grasp_data/meshes/{category}/{category}{i:03}.stl' if category in ["box","cylinder"] else f'grasp_data/meshes/{category}/{category}{i:03}.obj'    
	metadata_filename = f'grasp_data/info/{category}/{category}{i:03}.json'
	metadata = json.load(open(metadata_filename,'r'))


	# define grasp sampler
	graspsampler = GraspSampler(seed=1)

	# load object
	graspsampler.update_object(obj_filename=obj_filename, name=obj_filename, obj_scale=metadata['scale'])
	obj_pose_relative = graspsampler.obj.get_obj_mesh_mean()

	start_time = time.time()

	# get nearby grasps
	for q, t, index_grasp in tqdm( zip(quaternions,translations,range(len(quaternions)))):
		transform = qt2mat(q,t)
		new_quaternions = []
		new_translations = []

		transforms = utils.perturb_transform(transform, number_of_nearby_grasps,
			min_translation=(-0.01,-0.01,-0.01),
			max_translation=(0.01,0.01,0.01),
			min_rotation=(-0.125,-0.125,-0.125),
			max_rotation=(+0.125,+0.125,+0.125))



		is_promising = np.ones(100)
		# is_promising[qualities > 0] = 1  

		for _transform in transforms:
			_q,_t = mat2qt(_transform)
			new_quaternions.append(_q)
			new_translations.append(_t)
		print("size: ", np.array(new_quaternions).shape)
		filename = f"{save_dir}/{index_grasp:08}"
		np.savez_compressed(filename, transfroms=transforms, quaternions=new_quaternions, translations=new_translations,is_promising=is_promising,obj_pose_relative=obj_pose_relative)







from joblib import Parallel, delayed
Parallel(n_jobs=num_objects)(delayed(process)(i) for i in range(num_objects))

# process(7)