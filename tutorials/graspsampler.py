#!/usr/bin/python3.8

# Author: Tasbolat Taunyazov & Patrick Eala
# TUTORIAL 4: Graspsampler and its functions

import numpy as np
from graspsampler.common import PandaGripper, Scene, Object
import matplotlib.pyplot as plt
import trimesh.transformations as tra

from graspsampler.GraspSampler import GraspSampler

# init graspsampler constructor
graspsampler = GraspSampler(seed=11)
# load object
graspsampler.update_object(obj_filename='assets/sample_files/box000.stl', name='box')

#### generate samples
# number_of_grasps = 1
# points, normals, transforms, roll_angles, standoffs, collisions, qualities_1, qualities_2, type_of_qualities = graspsampler.sample_grasps(number_of_grasps=number_of_grasps, silent=True)

# # visualize grasps
# for i in range(number_of_grasps):
#     if (qualities_1[i] < 0.4) or (qualities_2[i] < 0.4):
#         continue
#     print('Sample ', i)
#     print('quality ', qualities_1[i], qualities_2[i])
#     graspsampler.grasp_visualize(transform=transforms[i],
#                         coordinate_frame=True,
#                         grasp_debug_items=True,
#                         other_debug_items=True,
#                         point=points[i])


# perturb grasp
number_of_grasps = 10
points, normals, transforms, origins, quaternions, alpha, beta, gamma, standoffs, qualities =\
     graspsampler.sample_grasps(number_of_grasps=number_of_grasps, silent=True)

# points, normals, transforms, roll_angles, standoffs, collisions, qualities_1, qualities_2, type_of_qualities =\

print(transforms[0])
print(qualities)
# graspsampler.grasp_visualize(transform=transforms[0],
#                         coordinate_frame=True,
#                         grasp_debug_items=True,
#                         other_debug_items=True,
#                         point=points[0])

# exit()

# points, normals, transforms, roll_angles, standoffs, collisions, qualities_1, qualities_2, type_of_qualities = \
    # graspsampler.perturb_grasp(points[0], normals[0], standoffs[0], roll_angles[0], 2,0.05,0.1)

# # visualize grasps
# for i in range(number_of_grasps):
#     print('Sample ', i)
#     print('quality ', qualities[i])
#     # print('quality ', qualities[i], qualities_2[i])
#     print(transforms[i])
#     graspsampler.grasp_visualize(transform=transforms[i],
#                         coordinate_frame=True,
#                         grasp_debug_items=True,
#                         other_debug_items=True,
#                         point=points[i])




# # perturb transform
# from graspsampler.utils import trans_matrix

# transform = trans_matrix(euler=[0,0,0], translation=[0,0,-0.15])

# graspsampler.grasp_visualize(transform=transform,
#                         coordinate_frame=True,
#                         grasp_debug_items=True,
#                         other_debug_items=False)

# transforms = graspsampler.perturb_transform(transform,10)

# for transform in transforms:
#     graspsampler.grasp_visualize(transform=transform,
#                         coordinate_frame=True,
#                         grasp_debug_items=True,
#                         other_debug_items=False)


# generate object point cloud
# full point cloud (make full_pc=False to get partial pc)
# pc, transferred_pose = graspsampler.get_pc(full_pc=True)
pc, transferred_pose, camera_pose = graspsampler.get_pc(full_pc=True)

graspsampler.view_pc(pc)
# graspsampler.update_object(obj_filename='assets/sample_files/box000.stl', name='box')
cat_id = "03261776"
# id = "7a4619d2240ac470620d74c38ad3f68f"
category = "earphone"
scale = 0.5
# path = f'shapenet/ShapeNetCore.v2/{cat_id}/{id}/models/model_normalized.obj'

# record =False
# check =True

# record =True
# check =False





import pickle

# db = pickle.load(open('models_sampled.pkl', 'rb'), encoding='latin1')

# for id in db.keys():
#     cat = db[id]["category"]    
#     print(cat)

# exit()



import os
dir = f'shapenet/ShapeNetCore.v2/{cat_id}/'

for id in os.listdir(dir):
    print(id)
    path = f'shapenet/ShapeNetCore.v2/{cat_id}/{id}/models/model_normalized.obj'
    graspsampler.update_object(obj_filename=path, name=category,obj_scale=scale)



    # perturb grasp
    number_of_grasps = 1
    points, normals, transforms, roll_angles, standoffs, collisions, qualities_1, qualities_2, type_of_qualities =\
        graspsampler.sample_grasps(number_of_grasps=number_of_grasps, silent=True)

    # print(transforms[0])
    graspsampler.grasp_visualize(transform=transforms[0],
                            coordinate_frame=True,
                            grasp_debug_items=True,
                            other_debug_items=True,
                            point=points[0])

    # points, normals, transforms, roll_angles, standoffs, collisions, qualities_1, qualities_2, type_of_qualities = \
    #     graspsampler.perturb_grasp(points[0], normals[0], standoffs[0], roll_angles[0], 2,0.05,0.1)

    # # # visualize grasps
    # for i in range(number_of_grasps):
    #     print('Sample ', i)
    #     print('quality ', qualities_1[i], qualities_2[i])
    #     print(transforms[i])
    #     graspsampler.grasp_visualize(transform=transforms[i],
    #                         coordinate_frame=True,
    #                         grasp_debug_items=True,
    #                         other_debug_items=True,
    #                         point=points[i])

    object_pass = input("Select object? (y/n): ")
    if object_pass == "y":
        pass
    elif object_pass == 'n':
        continue
    

    while True:
        print(f"Current scale: {scale}")
        scale = float(input(f"Enter new scale: "))
        graspsampler.update_object(obj_filename=path, name=category,obj_scale=scale)

        number_of_grasps = 1
        points, normals, transforms, roll_angles, standoffs, collisions, qualities_1, qualities_2, type_of_qualities =\
            graspsampler.sample_grasps(number_of_grasps=number_of_grasps, silent=True)

        graspsampler.grasp_visualize(transform=transforms[0],
                                coordinate_frame=True,
                                grasp_debug_items=True,
                                other_debug_items=True,
                                point=points[0])


        record = input("Is scale good? (y/n): ")

        if record == "y":
            break
        elif record == 'n':
            continue
        
    import pickle 
    db = pickle.load(open('models_sampled.pkl', 'rb'), encoding='latin1')
    model = {
             "id":id,
             "category":category,
             "path":path,
             "scale":scale
             }
    # del db[id]         
    db[id] = model
    print(db[id])
    # print(len(db.keys()))

    for id in db.keys():
        cat = db[id]["category"]    
        print(cat)
    with open('models_sampled.pkl','wb') as f:
        pickle.dump(db,f,protocol=2)

exit()

graspsampler.update_object(obj_filename=path, name=category,obj_scale=scale)

### generate samples
if not record and not check:
    number_of_grasps = 5000
    points, normals, transforms, roll_angles, standoffs, collisions, qualities_1, qualities_2, type_of_qualities = graspsampler.sample_grasps(number_of_grasps=number_of_grasps, silent=False)

    # visualize grasps
    for i in range(number_of_grasps):
        if (qualities_1[i] < 0.2) or (qualities_2[i] < 0.2):
            continue
        print('Sample ', i)
        print('quality ', qualities_1[i], qualities_2[i])
        print('transformation', transforms[i])
        graspsampler.grasp_visualize(transform=transforms[i],
                            coordinate_frame=True,
                            grasp_debug_items=True,
                            other_debug_items=True,
                            point=points[i])

if check:
    # perturb grasp
    number_of_grasps = 1
    points, normals, transforms, roll_angles, standoffs, collisions, qualities_1, qualities_2, type_of_qualities =\
        graspsampler.sample_grasps(number_of_grasps=number_of_grasps, silent=True)

    print(transforms[0])
    graspsampler.grasp_visualize(transform=transforms[0],
                            coordinate_frame=True,
                            grasp_debug_items=True,
                            other_debug_items=True,
                            point=points[0])

    points, normals, transforms, roll_angles, standoffs, collisions, qualities_1, qualities_2, type_of_qualities = \
        graspsampler.perturb_grasp(points[0], normals[0], standoffs[0], roll_angles[0], 2,0.05,0.1)

    # # visualize grasps
    for i in range(number_of_grasps):
        print('Sample ', i)
        print('quality ', qualities_1[i], qualities_2[i])
        print(transforms[i])
        graspsampler.grasp_visualize(transform=transforms[i],
                            coordinate_frame=True,
                            grasp_debug_items=True,
                            other_debug_items=True,
                            point=points[i])

    #perturb transform
    from graspsampler.utils import trans_matrix

    transform = trans_matrix(euler=[0,0,0], translation=[0,0,-0.15])

    graspsampler.grasp_visualize(transform=transform,
                            coordinate_frame=True,
                            grasp_debug_items=True,
                            other_debug_items=False)

    transforms = graspsampler.perturb_transform(transform,10)

    for transform in transforms:
        graspsampler.grasp_visualize(transform=transform,
                            coordinate_frame=True,
                            grasp_debug_items=True,
                            other_debug_items=False)

if record:
    import pickle 
    db = pickle.load(open('models_sampled.pkl', 'rb'), encoding='latin1')
    model = {
             "id":id,
             "category":category,
             "path":path,
             "scale":scale
             }
    # del db[id]         
    db[id] = model
    # print(db[id])
    # print(len(db.keys()))

    for id in db.keys():
        cat = db[id]["category"]    
        print(cat)
    with open('models_sampled.pkl','wb') as f:
        pickle.dump(db,f,protocol=2)


# multiple partial point clouds
pcs, transferred_poses = graspsampler.get_multiple_random_pcs(number_of_pcs=3)
for pc in pcs:
    graspsampler.view_pc(pc)
