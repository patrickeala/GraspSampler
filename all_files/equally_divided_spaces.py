from graspsampler.GraspSampler import GraspSampler
import json

trial = 2
category = 'mug'

i = 0
obj_filename = f'grasp_data/meshes/{category}/{category}{i:03}.obj'    
metadata_filename = f'grasp_data/info/{category}/{category}{i:03}.json'
metadata = json.load(open(metadata_filename,'r'))

# define grasp sampler
graspsampler = GraspSampler(seed=trial)

# load object
graspsampler.update_object(obj_filename=obj_filename, name=obj_filename, obj_scale=metadata['scale'])

quaternions, translations, transforms, qualities = graspsampler.sample_equal_spaced_grasps(silent=True)

print(quaternions.shape)