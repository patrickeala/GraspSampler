#!/usr/bin/python3.8

# Author: Tasbolat Taunyazov
# TUTORIAL 1: Defining gripper and visualizing it.

import numpy as np
from graspsampler.common import PandaGripper, Scene, Object
import matplotlib.pyplot as plt
import trimesh.transformations as tra

print('---------------------Gripper definition tutorial-----------------')

# define gripper
root_folder = ''
gripper = PandaGripper(root_folder=root_folder)

scene = Scene() # create a Scene
scene.add_coordinate_frame() # add coordinate frame for debugging
scene.add_gripper(gripper) # add gripper
scene.add_gripper_bb(gripper) # add gripper's bounding box
scene.add_rays(gripper.get_closing_rays()) # visualize rays

scene.show() # visualize the scene

print('Each created geometry adds a node to the scene. Remove them if gripper is rotated & translated.')

# NOTE: remove rays and bounding boxes before transformation
scene.remove_rays(gripper) # remove rays
scene.remove_gripper_bb(gripper) # remove bbs
scene.show() # visualize to see the effect

print('Now gripper can be rotated to -90 degreeze along Z')

# rotate the gripper
trans = tra.euler_matrix(0, 0, -np.pi/6)
gripper.apply_transformation(trans) # rotate the gripper

scene.add_rays(gripper.get_closing_rays(trans)) # do not forget add transformation to rays too!
scene.add_gripper_bb(gripper) # see how bb looks like
scene.show() # visualize


print('NOTE: if gripper is rotated & translated with bb and rays on the scene, \
      these components are not affected. \
      Make sure to remove them before visualizing.')
