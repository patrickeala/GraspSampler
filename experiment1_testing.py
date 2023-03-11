import torch
import numpy as np
import argparse
# from models.models import GraspSamplerDecoder, GraspEvaluator
# from models.quaternion import quaternion_mult
# from utils import density_utils
import pickle
import time
from scipy.spatial.transform import Rotation as R
from graspsampler import utils
from graspsampler.common import PandaGripper, Scene

info, args = pickle.load(open('results/1641738054/info', 'rb'))
N = 50
success_last = info['success'][-1, :]

# correct success labels : 237 752 931 869 812
    # nonstable but success : 202
    # collision : 31 875 725 341 406 882 724 150 726 549
    # collision (sensitive) : 802 407 875 725 993 239 554 218 570 748 947
    # far : 504 747 548 248 758 (285) 831 (265) 588 818 (523) 471 525 193 979 773 255
    # abnormal : 801 938 673 631 309 23 362 872

data_types = {
    'correct_success_labels': [237, 202, 752, 801, 938, 931, 673, 869, 812, 309, 193, 362, 872],
    'collision': [31, 875, 725, 341, 406, 882, 724, 150, 726, 549],
    'collision_sensitive': [802, 407, 875, 725, 993, 239, 554, 218, 570, 748, 947],
    'far': [504, 747, 548, 248, 758, 285, 831, 265, 588, 818, 523, 471, 525, 979, 773, 255],
    'abnormal': [631, 23],
}

all_indices  = sum(data_types.values(), [])
number_of_grasps = 100
all_translations = []
all_quaternions = []
for mask in all_indices:
    translation= info['translations'][-1, mask, :]
    quaternion= info['quaternions'][-1, mask, :]
    transform = np.eye(4)
    transform[:3,3] = translation
    transform[:3,:3] = R.from_quat(quaternion).as_matrix()
    transforms= utils.perturb_transform(transform, number_of_grasps, 
                    min_translation=(-0.01,-0.01,-0.01),
                    max_translation=(0.01,0.01,0.01),
                    min_rotation=(-0.125,-0.125,-0.125),
                    max_rotation=(+0.125,+0.125,+0.125))
    # transforms= utils.perturb_transform(transform, number_of_grasps, 
    #                 min_translation=(-0.001,-0.001,-0.001),
    #                 max_translation=(0.001,0.001,0.001),
    #                 min_rotation=(-0.0125,-0.0125,-0.0125),
    #                 max_rotation=(+0.0125,+0.0125,+0.0125))

    transforms = np.asarray(transforms)
    _translations = transforms[:,:3,3]
    all_translations.append(_translations)
    _quaternions = R.from_matrix(transforms[:, :3, :3]).as_quat()
    all_quaternions.append(_quaternions)

    #data[mask] = [_translations, _quaternions]
all_translations = np.vstack(all_translations)
all_quaternions = np.vstack(all_quaternions)

print(all_translations.shape, all_quaternions.shape)

pickle.dump([all_translations, all_quaternions,all_indices, data_types], open('experiment1_test.pkl', 'wb'))