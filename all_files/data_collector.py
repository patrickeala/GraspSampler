from random import sample
import numpy as np
import pickle
from graspsampler.common import PandaGripper, Scene, Object
import time
import json
from os.path import exists
from pathlib import Path
from graspsampler.GraspSampler import GraspSampler



cat = "spatula"


info_save_dir = f"grasp_data/info/{cat}"
Path(info_save_dir).mkdir(parents=True, exist_ok=True)


for idx in range(20):
    info_file = f"{info_save_dir}/{cat}{idx:03}.json"
    object_filename = f"grasp_data/meshes/{cat}/{cat}{idx:03}.obj"

    if exists(info_file):
        continue

    while True:
        obj_scale_x = float(input("input the object scale x: "))
        obj_scale_y = float(input("input the object scale y: "))
        obj_scale_z = float(input("input the object scale z: "))
        scale = [obj_scale_x,obj_scale_y,obj_scale_z]
        sampler = GraspSampler()
        sampler.update_object(obj_filename=object_filename, name=object_filename, obj_scale=scale)
        transform = np.eye(4)
        sampler.grasp_visualize(transform, coordinate_frame=True,grasp_debug_items=True,other_debug_items=False)

        # object_instance = Object(filename=object_filename,name=cat,scale=scale, shift_to_center=True)
        # scene = Scene()
        # gripper = PandaGripper()
        # scene.add_object(object_instance)
        # scene.add_gripper(gripper)

        # scene.show()
        object_selected = input("input 1 is selected else 2")
        if int(object_selected) == 1:
            dictionary = {"id":idx,"category":cat,"path":"3DNet","scale":scale}
            json_object = json.dumps(dictionary)
            with open(info_file, "w") as outfile:
                outfile.write(json_object)
            break









