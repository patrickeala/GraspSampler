
# from experiment_utils import utils

# import torch
import numpy as np
import argparse
# from models.models import GraspSamplerDecoder, GraspEvaluator
# from models.quaternion import quaternion_mult
# from utils import density_utils
import pickle
from graspsampler.GraspSampler import GraspSampler
from graspsampler.common import PandaGripper, Scene, Object
# from torch.utils.data import DataLoader, TensorDataset
# import torch.nn.functional as F
# from tqdm.auto import tqdm
# from pathlib import Path
# import time
from scipy.spatial.transform import Rotation as R
from graspsampler import utils
from graspsampler.common import PandaGripper, Scene
import trimesh


panda_gripper = PandaGripper()
p = trimesh.util.concatenate(panda_gripper.get_meshes())
p.visual.face_colors = [100,100,100,100]
scene = trimesh.Scene()
scene.add_geometry(p)
scene.add_geometry(utils.gripper_bd(0))
scene.show()