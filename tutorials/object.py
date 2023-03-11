#!/usr/bin/python3.8

# Author: Tasbolat Taunyazov
# TUTORIAL 2: Defining object and visualizing it.

import numpy as np
from graspsampler.common import PandaGripper, Scene, Object
import trimesh.transformations as tra

# define object
box = Object(filename='assets/sample_files/box000.stl', name='box')

# define gripper
root_folder = ''
gripper = PandaGripper(root_folder=root_folder)

scene = Scene() # create scene
# scene.add_coordinate_frame() # add coordinate frame
scene.add_gripper(gripper) # add gripper
scene.add_object(box, face_colors=[0, 255, 0, 200]) # add box
scene.show()

box.apply_transform(tra.euler_matrix(0,0,-np.pi/6)) # rotate object -90 degree around Z.

# collision checking
print('Is object in collision with gripper?: expected True')
print( box.in_collision_with(gripper.base ))

scene.show() # visualize scene




box.apply_transform(tra.translation_matrix([0,0,0.2])) # rotate object -90 degree around Z.

# collision checking.
print('Is object in collision with gripper?: expected False')
print( box.in_collision_with(gripper.base, tra.euler_matrix(0,0,-np.pi/2)) )

scene.show()


