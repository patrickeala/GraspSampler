import numpy as np
import pickle
from graspsampler.GraspSampler import GraspSampler
import time
import json
import os
from pathlib import Path
from graspsampler.common import PandaGripper, Scene, Object
from graspsampler.GraspSampler import GraspSampler
from graspsampler.utils import gripper_bd
import trimesh
import open3d as o3d
from trimesh.transformations import transform_points


# define grasp sampler
NUM_PCS = 3
TARGET_PC_SIZE  = None



def sample_grasps(filename,path,name,scale,path_grasps,path_depths):
    graspsampler = GraspSampler(seed=10)
  
    # load object
    graspsampler.update_object(obj_filename=path, name=name, obj_scale=scale)
    # obj_pose_relative = graspsampler.obj.get_obj_mesh_mean()
    # # sample grasps:
    # _, normals, transforms, origins, quaternions,\
    #     alpha, beta, gamma, _, collisions, qualities_1, qualities_2 = graspsampler.sample_grasps_test(number_of_grasps=NUM_GRASPS,
    #                                                                              alpha_lim = [0, 2*np.pi],
    #                                                                              beta_lim = [-np.pi, np.pi],
    #                                                                              gamma_lim = [-np.pi, np.pi],
    #                                                                              silent=False)
    # # explore locally
    # promising_candidates = (qualities_1 >0) & (qualities_2 >0)
    # number_of_promising_candts = len(transforms[promising_candidates])

    # if number_of_promising_candts != 0:
    #     new_qualities_1, new_qualities_2, new_origins, new_quaternions, new_transforms  = graspsampler.perturb_grasp_locally_test(
    #                                                                             number_of_grasps=NUM_LOCAL_GRASPS,
    #                                                                             qualities_1 = qualities_1[promising_candidates],
    #                                                                             normals=normals[promising_candidates],
    #                                                                             origins=origins[promising_candidates],
    #                                                                             alpha=alpha[promising_candidates],
    #                                                                             beta=beta[promising_candidates],
    #                                                                             gamma=gamma[promising_candidates],
    #                                                                             alpha_range=np.pi/36*4,
    #                                                                             beta_range=np.pi/18*4,
    #                                                                             gamma_range=np.pi/18*4,
    #                                                                             t_range=0.05, # 0.005
    #                                                                             recur_search_depth = recur_search_depth,
    #                                                                             silent=False)
    #     # new_qualities_1 
    #     # new_qualities_2
    #     transforms = np.concatenate([transforms, new_transforms])
    #     origins = np.concatenate([origins, new_origins])
    #     quaternions = np.concatenate([quaternions, new_quaternions])
    #     qualities_1 = np.concatenate([qualities_1, new_qualities_1])
    #     qualities_2 = np.concatenate([qualities_2, new_qualities_2])

    #     del new_transforms, new_origins, new_quaternions,  new_qualities_1, new_qualities_2

    # data_grasp = {}
    # data_grasp['fname'] = obj_filename
    # data_grasp['transforms'] = transforms
    # data_grasp['translations'] = origins
    # data_grasp['quaternions'] = quaternions
    # data_grasp['collisions'] = collisions
    # data_grasp['qualities_1'] = qualities_1
    # data_grasp['qualities_2'] = qualities_2
    # data_grasp['obj_pose_relative'] = obj_pose_relative
    # data_grasp['scale'] =scale
    # # add some labels for now only
    # promising_candidates = (qualities_1 >0) & (qualities_2 >0)
    # promising_candidates_count = len(transforms[promising_candidates])

    # with open(f'{path_grasps}/{filename}_depth_10_number_10000.pkl', 'wb') as fp:
    #     pickle.dump(data_grasp, fp, protocol=pickle.HIGHEST_PROTOCOL)
    

    # sample point clouds
    Path(path_depths).mkdir(parents=True, exist_ok=True)
    pcs,depths,transfered_poses, camera_poses = graspsampler.get_multiple_random_pcs(number_of_pcs=NUM_PCS, 
                                            target_pc_size=TARGET_PC_SIZE, 
                                            depth_noise=[0,0], 
                                            dropout=0)
            
    data = {}
    data["depth_maps"] = depths
    data["transfered_poses"] = transfered_poses
    # with open(f'{path_depths}/{filename}.pkl', 'wb') as fp:
    #     pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    # del pcs, depths, transfered_poses
    # total_grasps = len(data_grasp['qualities_1'])
    # print(f'Initial number of promising candidates: {number_of_promising_candts}')
    # print(f'New number of promising candidates: {promising_candidates_count}')
    # print(f'Total number of grasps gained: {total_grasps}')
    print(f"done wiht {name}")  
    print(camera_poses)
    return pcs,depths,transfered_poses, camera_poses
    # del data_grasp, scale, normals, transforms, origins, alpha, collisions, qualities_1, qualities_2
# for object in os.listdir(data_dir):
#     print(object)
#     if object != "box":
#         continue
#     path_grasps = os.path.join(grasp_save_dir,object)
#     path_pcs = os.path.join(pc_save_dir,object)
#     for sample in os.listdir(f'{data_dir}/{object}'):
#         filename = f'{data_dir}/{object}/{sample}'
#         with open(filename) as json_file:
#             data = json.load(json_file)
#         name = data['id']
#         path = data['path']
#         if not path:
#             path = f'grasp_data/meshes/{object}/{object}{sample[-8:-5]}.stl'
#         scale = data["scale"]
#         target_filename = sample.split(".")[0]
#         print("target_filename: ", target_filename)
#         print("path: ", path)
# for cat in ["scissor","fork","hammer"]:
# for cat in ["fork","hammer"]:
for cat in ["fork"]:

    for idx in range(5):
        # cat = "scissor" 
        path = f"grasp_data/meshes/{cat}/{cat}{idx:03}.obj"
        path_pcs = f"grasp_data/pcls/{cat}"
        dtype = o3d.core.float32

        # path_depths = f"depth_maps/ycb/"
        metadata_filename = f'grasp_data/info/{cat}/{cat}{idx:03}.json'
        metadata = json.load(open(metadata_filename,'r'))
        pcs = np.array(sample_grasps(cat,path,name=cat,scale=metadata["scale"],path_grasps=None,path_depths=None))
        pcs_ = pcs[0]
        print(pcs_.shape)
        for _pc in pcs:
            pcs_ = np.concatenate([pcs_,_pc])
        pc_target = pcs_

        Path(path_pcs).mkdir(parents=True, exist_ok=True)

        with open(f'{path_pcs}/{cat}{idx:03}.pkl', 'wb') as fp:
            pickle.dump(pcs_, fp, protocol=3)
        del pcs




# obj = "024_bowl" 
# path = f"assets/meshes_10_objects/{obj}/google_16k/textured.obj"
# path_depths = f"depth_maps/ycb/"

# sample_grasps(obj,path,name=obj,scale=1,path_grasps=None,path_depths=path_depths)


# def process(i):
#     obj = f"{cat}{i:03}"
#     path_depths = f"depth_maps/training_data/{cat}"
#     path = f"grasp_data/meshes/{cat}/{obj}.obj"
#     if cat in ["box","cylinder"]:
#         path = f"grasp_data/meshes/{cat}/{obj}.stl"
#     metadata_filename = f"grasp_data/info/{cat}/{obj}.json"
#     metadata = json.load(open(metadata_filename,"r"))

#     sample_grasps(obj,path,name=obj,scale=metadata['scale'],path_grasps=None,path_depths=path_depths)

# from joblib import Parallel, delayed
# Parallel(n_jobs=21)(delayed(process)(i) for i in range(21))

cat = "mug"
idx = 4
obj =f"{cat}{idx:03}"
path = f"grasp_data/meshes/{cat}/{obj}.obj"
if cat in ["box","cylinder"]:
    path = f"grasp_data/meshes/{cat}/{obj}.stl"
path_depths = f"debug_folder"
metadata_filename = f"grasp_data/info/{cat}/{obj}.json"
metadata = json.load(open(metadata_filename,"r"))
pcs, depths,transfered_poses, camera_poses = sample_grasps(obj,path,name=obj,scale=metadata['scale'],path_grasps=None,path_depths=path_depths)
# for i in range(len(pcs)):
#     pcs[i] = np.matmul(pcs[i],camera_poses[i])
pcs_ = pcs[0]
extra = np.array([[-1.0000000,  0.0000000,  0.0000000, 0],
                  [ 0.0000000,  1.0000000,  0.0000000, 0],
                  [ 0.0000000,  0.0000000, -1.0000000, 0],
                  [ 0.0000000,  0.0000000, 0, 1]])

pcs_ = transform_points(pcs_, camera_poses[0])
pcs_ = transform_points(pcs_, extra)

# pcs_ = 
# for _pc in pcs:
#     pcs_ = np.concatenate([pcs_,_pc])
pc1 = trimesh.points.PointCloud(vertices=pcs_)
# pc1.apply_transform((camera_poses[0]))
# scene = Scene()
# scene.add_geometry(pc1)
graspsampler = GraspSampler(seed=10)
graspsampler.update_object(obj_filename=path, name=obj, obj_scale=metadata['scale'])
graspsampler.scene.add_object(graspsampler.obj)
graspsampler.scene.add_geometry(pc1)
# graspsampler.view_pc(pcs[0])
graspsampler.scene.show()