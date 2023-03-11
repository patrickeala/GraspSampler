import numpy as np
import pickle
import json
import os
import shutil

data_dir = "training_data_2"

def copy_and_rename(target_filename,obj_file,target_path):
    target_file = f"{target_path}/{target_filename}.obj"
    shutil.copy(obj_file,target_file)
    print(f"done with {target_filename} form {obj_file} to {target_file}")
    
for object in os.listdir(data_dir):
    target_path = f"/home/user/isaacgym/python/IsaacGymGrasp/assets/{object}"
    for sample in os.listdir(f'{data_dir}/{object}'):
        filename = f'{data_dir}/{object}/{sample}'
        with open(filename) as json_file:
            data = json.load(json_file)
        path = data['path']
        target_filename = sample.split(".")[0]
        copy_and_rename(target_filename,path,target_path)
