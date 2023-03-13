
# from experiment_utils import utils

# import torch
from gettext import translation
from hashlib import new
import numpy as np
from graspsampler.GraspSampler import GraspSampler
from graspsampler.common import PandaGripper, Scene, Object
import json
from scipy.spatial.transform import Rotation as R
from graspsampler import utils
from graspsampler.common import PandaGripper, Scene
import trimesh
    
# cat = "mug"
# idx = 2
def relabel(cat,idx):
    data_path = f"/home/user/isaac/grasps_sampled_for_positive_grasps/{cat}/{cat}{idx:03}_isaac_lower_friction_with_num_intersection_rays"
    for file_idx in range(1000):
        grasp_data_filename = f"{data_path}/{file_idx:08}.npz"
        grasp_data = np.load(grasp_data_filename)
 
        isaac_labels=grasp_data["isaac_labels"]
        print(grasp_data_filename)
        num_intersection_rays=grasp_data["num_intersection_rays"]

        num_rays_stable_grasps = []
        for i, n in zip(isaac_labels, num_intersection_rays):
            if i == 1 and n != 0:
                num_rays_stable_grasps.append(n)

        if num_rays_stable_grasps == []:
            new_labels = np.array(isaac_labels)
            np.savez_compressed(file=grasp_data_filename,quaternions=grasp_data["quaternions"], translations=grasp_data["translations"],
            isaac_labels=grasp_data["isaac_labels"],new_labels=np.array(new_labels))
            continue

        threshold_mean = np.average(num_rays_stable_grasps) // 1
        threshold_median = np.median(num_rays_stable_grasps) // 1
        threshold_minimum = np.min(num_rays_stable_grasps)
        print(sum(num_rays_stable_grasps), len(num_rays_stable_grasps))
        print("average threshold is : ", threshold_mean)
        print("median threshold is : ", threshold_median)
        print("minimum threshold is : ", threshold_minimum)

        threshold = threshold_median
        new_labels = []
        for n, l in zip(num_intersection_rays, isaac_labels):  
            if l == 1:
                new_labels.append(1)
                continue
            if n >= threshold:
                new_labels.append(1)
            else:
                new_labels.append(0)
        assert(len(new_labels)==len(isaac_labels))

        np.savez_compressed(file=grasp_data_filename,quaternions=grasp_data["quaternions"], translations=grasp_data["translations"],
        isaac_labels=grasp_data["isaac_labels"],new_labels=np.array(new_labels))


for cat in ["bowl"]:
    for idx in range(1):
        relabel(cat,idx)


exit()
#  # quaternions=grasp_data["quaternions"]
#         # translations=grasp_data["translations"]
#         # # Get object mesh
#         # obj_filename = f"grasp_data/meshes/{cat}/{cat}{idx:03}.stl" if cat in ["box","cylinder"] else  f"grasp_data/meshes/{cat}/{cat}{idx:03}.obj"
#         # metadata_filename = f'grasp_data/info/{cat}/{cat}{idx:03}.json'
#         # metadata = json.load(open(metadata_filename,'r'))
#         # sampler = GraspSampler()
#         # sampler.update_object(obj_filename=obj_filename, name=obj_filename, obj_scale=metadata['scale'])


#         # obj = Object(obj_filename, cat, scale=metadata['scale'])
#         # scene = Scene()
#         # scene.add_geometry(obj.mesh, geom_name='object')
# # counter = 0
# # /


#     # except:
#     #     continueata = np.load(f'{data_path}/main_grasps.npz')
    
#     # main_grasps_data = np.load(f'{data_path}/main_grasps.npz')
#     # main_grasp_quaternion = main_grasps_data["quaternions"][grasp_idx]
#     # main_grasp_translation = main_grasps_data["translations"][grasp_idx]

#     # sub_grasps_data = np.load(f'{data_path}/{grasp_idx:08}.npz')
#     # sub_grasps_quaternions = sub_grasps_data["quaternions"]
#     # sub_grasps_translations = sub_grasps_data["translations"]
#     # sub_grasps_labels = sub_grasps_data["isaac_labels"]

#     quaternion = quaternions[grasp_idx]
#     translation = translations[grasp_idx]
#     # print(quaternion)
#     # print(translation)
#     # continue

#     # add main grasp
#     # if isaac_fake_labels[grasp_idx] == 0:
#         # continue
#     gripper = utils.gripper_bd(quality=isaac_fake_labels[grasp_idx], opacity=0.75)
#     r = R.from_quat(quaternion)
#     transform = np.eye(4)
#     transform[:3,:3] = r.as_matrix()
#     transform[:3,3] = translation
#     gripper.apply_transform(transform)
#     scene.a
# for q, t, n, l in zip(quaternions, translations, num_intersection_rays, isaac_labels):
#     if n >= threshold and l == 0:
#         gripper = utils.gripper_bd()
#         r = R.from_quat(q)
#         transform =     # except:
#     #     continue

# print(f"got {counter} new positive grasps")
# scene.show()

# dd_geometry(gripper, geom_name='gripper')
#     # scene.show()
#     # scene.delete_geometry('gripper')

#     # for i in range(100):
#     #     # if sub_grasps_labels[i] == 1:
#     #     #     continue
#     #     gripper = utils.gripper_bd(sub_grasps_labels[i])
#     #     r = R.from_quat(sub_grasps_quaternions[i])
#     #     transform = np.eye(4)
#     #     transform[:3,:3] = r.as_matrix()
#     #     transform[:3,3] = sub_grasps_translations[i]
#     #     gripper.apply_transform(transform)
#     #     scene.add_geometry(gripper)
        
# scene.show()

# # exit()
# # scene.add_geometry(obj_mesh)





# # number_of_grasps = 50
# # panda = PandaGripper()
# # panda.apply_transformation(transform)
# # scene.add_gripper(panda)


# # for i in range(translations_final.shape[0]):
    
# #     gripper = utils.gripper_bd(1)
# #     r = R.from_quat(quaternions_final[i])
# #     transform = np.eye(4)
# #     transform[:3,:3] = r.as_matrix()
# #     transform[:3,3] = translations_final[i]

# #     gripper.apply_transform(transform)
# #     scene.add_geometry(gripper)

# for q, t, n, l in zip(quaternions, translations, num_intersection_rays, isaac_labels):
#     if n >= threshold and l == 0:
#         gripper = utils.gripper_bd()
#         r = R.from_quat(q)
#         transform =     # except:
#     #     continue

# print(f"got {counter} new positive grasps")
# scene.show()


#     # transforms= utils.perturb_transform(transform, number_of_grasps, 
#     #                 min_translation=(-0.01,-0.01,-0.01),
#     #                 max_translation=(0.01,0.01,0.01),
#     #                 min_rotation=(-0.125,-0.125,-0.125),
#     #                 max_rotation=(+0.125,+0.125,+0.125))
#     # print(len(transforms))

#     # for _trans in transforms:
#     #     gripper = utils.gripper_bd(0)
#     #     gripper.apply_transform(_trans)
#     #     scene.add_geometry(gripper)


# # for i in range(all_trans.shape[1]):
# #     gripper = utils.gripper_bd(1)
# #     r = R.from_euler('XYZ', all_eulers[-1][i])
# #     transform = np.eye(4)
# #     transform[:3,:3] = r.as_matrix()
# #     transform[:3,3] = all_trans[-1][i]
# #     gripper.apply_transform(transform)
# #     scene.add_geometry(gripper)

# for q, t, n, l in zip(quaternions, translations, num_intersection_rays, isaac_labels):
#     if n >= threshold and l == 0:
#         gripper = utils.gripper_bd()
#         r = R.from_quat(q)
#         transform =     # except:
#     #     continue

# print(f"got {counter} new positive grasps")
# scene.show()


# # all_trans_v = np.mean(all_trans_v, axis=1)
# # print(all_trans_v.shape)
# # plt.plot(all_trans_v[:,0], label='x')
# # plt.plot(all_trans_v[:,1], label='y')
# # plt.plot(all_trans_v[:,2], label='z')
# # # plt.show()

# # # all_quat_v = all_quat_v.squeeze(1)
# # # plt.plot(all_quat_v[:,0], 'r--')
# # # plt.plot(all_quat_v[:,1], 'r--')
# # # plt.plot(all_quat_v[:,2], 'r--')
# # # plt.plot(all_quat_v[:,3], 'r--')
# # # plt.show()

# # all_euler_v = np.mean(all_euler_v, axis=1)
# # plt.plot(all_euler_v[:,0], 'r--')
# # plt.plot(all_euler_v[:,1], 'r--')
# # plt.plot(all_euler_v[:,2], 'r--')
# # plt.legend()
# # plt.show()



# # all_success = all_success.squeeze()
# # print(all_success.shape)
# # plt.plot(all_success)
# # plt.show()