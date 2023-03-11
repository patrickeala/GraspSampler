from numpy.random.mtrand import rand
from trimesh.permutate import transform
from graspsampler.common import PandaGripper, Scene, Object
from graspsampler.GraspSampler import GraspSampler
from graspsampler.utils import gripper_bd
import trimesh
import pickle
import numpy as np
import json
import torch
from scipy.spatial.transform import Rotation as R

i = 8
category = 'bowl'
trial = 1
# load object and control points

save_dir = 'grasp_data_generated'
obj_filename = f'grasp_data/meshes/{category}/{category}{i:03}.obj'    
metadata_filename = f'grasp_data/info/{category}/{category}{i:03}.json'
metadata = json.load(open(metadata_filename,'r'))
obj = Object(filename=obj_filename, name=obj_filename, scale=metadata['scale'])
obj.mesh.visual.face_colors = [255,255,255,25]
graspsampler = GraspSampler(seed=trial)

# load object
graspsampler.update_object(obj_filename=obj_filename, name=obj_filename, obj_scale=metadata['scale'])
pcs = graspsampler.get_multiple_random_pcs(2, depth_noise=0.0, dropout=0.0)[0]
pc_mean = pcs.mean(axis=0)
print(pc_mean)

pc1 = trimesh.points.PointCloud(vertices=pcs)
pc2 = trimesh.points.PointCloud(vertices=(pcs-pc_mean))
scene = Scene()
#scene.add_coordinate_frame()
scene.add_object(obj)
scene.add_geometry(pc1)
#scene.add_geometry(pc2)
data = np.load( f'{save_dir}/{category}/{category}{i:03}/main{trial}.npz')

def quaternion_conj(q):
    """
      Conjugate of quaternion q (x,y,z,w) -> (-x,-y,-z,w).
    """
    q_conj = q.clone()
    q_conj[:, :, :3] *= -1
    return q_conj

def quaternion_mult(q, r):
    """
    Multiply quaternion(s) q (x,y,z,w) with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    x =terms[:, 3, 0] + terms[:, 2, 1] - terms[:, 1, 2] + terms[:, 0, 3]
    y = - terms[:, 2, 0] + terms[:, 3, 1] + terms[:, 0, 2] + terms[:, 1, 3]
    z = terms[:, 1, 0] - terms[:, 0, 1] + terms[:, 3, 2] + terms[:, 2, 3]
    w = - terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] + terms[:, 3, 3] 
   

    return torch.stack((x, y, z, w), dim=1).view(original_shape)

def rot_p_by_quaterion(p, q):
    """
      Takes in points with shape of (batch_size x n x 3) and quaternions with
      shape of (batch_size x n x 4) and returns a tensor with shape of 
      (batch_size x n x 3) which is the rotation of the point with quaternion
      q. 
    """
    shape = p.shape
    q_shape = q.shape

    assert (len(shape) == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (shape[-1] == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (len(q_shape) == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (q_shape[-1] == 4), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (q_shape[1] == shape[1]), 'point shape = {} q shape = {}'.format(
        shape, q_shape)

    q_conj = quaternion_conj(q)
    r = torch.cat([ p,
        torch.zeros(
            (shape[0], shape[1], 1), dtype=p.dtype).to(p.device)],
                  dim=-1)
    result = quaternion_mult(quaternion_mult(q, r), q_conj)
    return result[:,:,:3] 

def transform_gripper_pc_old(quat, trans):
    # q: (x,y,z, w)
    # t: (x,y,z)
    
    # upload gripper_pc
    control_points = np.load('assets/gripper_control_points/panda.npy')[:, :3]
    control_points = [[0, 0, 0], [0, 0, 0], control_points[0, :],
                      control_points[1, :], control_points[-2, :],
                      control_points[-1, :]]
    control_points = np.asarray(control_points, dtype=np.float32)
    control_points = np.tile(np.expand_dims(control_points, 0),
                             [quat.shape[0], 1, 1])

    gripper_pc = torch.tensor(control_points).to(quat.device)  

    # prepare q and t 
    quat = quat.unsqueeze(1).repeat([1, gripper_pc.shape[1], 1])
    trans = trans.unsqueeze(1).repeat([1, gripper_pc.shape[1], 1])

    # rotate and add
    gripper_pc = rot_p_by_quaterion(gripper_pc, quat)
    gripper_pc +=trans

    return gripper_pc

def augment_grasp(pc, quaternion, translation):
    '''
    pc: [n, 3]
    quaternion: [4]
    translation: [3]
    '''
    # sample random unit quaternion
    rand_quat = torch.FloatTensor([[0.707,0.707,0,0.0]])

    # rotate pc
    pc = pc.unsqueeze(0)
    rand_quat1 = rand_quat.unsqueeze(1).repeat([1,pc.shape[1], 1])
    print(rand_quat1.shape)
    pc = rot_p_by_quaterion(pc, rand_quat1).squeeze()

    # rotate translation
    translation = translation.unsqueeze(0).unsqueeze(0)
    rand_quat2 = rand_quat.unsqueeze(1).repeat([1,1, 1])
    translation = rot_p_by_quaterion(translation, rand_quat2).squeeze()

    # rotate quaternion
    quaternion = quaternion.unsqueeze(0)
    quaternion = quaternion_mult(rand_quat, quaternion).squeeze()
    return pc, quaternion, translation

for i in range(0, len(data['transforms'])):

    #small_gripper = gripper_bd(data['is_promising'][i])
    box = trimesh.primitives.Sphere(radius=0.01)
    
    gripper = PandaGripper(root_folder='')
    gripper.apply_transformation(data['transforms'][i])
    scene.add_gripper(gripper)
    #scene.add_geometry(small_gripper, transform=data['transforms'][i])
    scene.add_geometry(box, transform=data['transforms'][i])
    # gripper.apply_transformation(data['transforms'][i])
    #scene.add_gripper(gripper)

    translation = data['translations'][i]
    #_transform = data['transforms'][i]

    #translation = translation - pc_mean
    #transform[:3,3] = translation
    #scene.add_geometry(box, transform=_transform)
    #gripper = PandaGripper(root_folder='')
    #gripper.apply_transformation(_transform)
    #scene.add_gripper(gripper)

    quaternion = torch.FloatTensor(data['quaternions'][i]).unsqueeze(0)
    translation = torch.FloatTensor(translation).unsqueeze(0)
    print(quaternion.shape, translation.shape)
    gripper_pc = transform_gripper_pc_old(quaternion, translation).squeeze(0)
    print(gripper_pc.shape)
    
    # for p in gripper_pc:
    #     b = trimesh.primitives.Sphere(radius=0.005)
    #     tt = np.eye(4)
    #     tt[:3,3] = p
    #     scene.add_geometry(b, transform=tt)


    # augmentation


    pc1 = torch.FloatTensor(pcs)
    quaternion = torch.FloatTensor(data['quaternions'][i])
    translation = torch.FloatTensor(data['translations'][i])

    pc, quaternion, translation = augment_grasp(pc1, quaternion, translation)
    pc, quaternion, translation = pc.numpy(), quaternion.numpy(), translation.numpy()

    # rand_quat = torch.FloatTensor([[0.707,0.707,0,0.0]])
    # print(rand_quat.shape)
    # rand_quat1 = rand_quat.unsqueeze(1).repeat([1,pc1.shape[1], 1])
    # print(pc1.shape, rand_quat1.shape)
    # pc1 = rot_p_by_quaterion(pc1, rand_quat1).squeeze()
    pc1 = trimesh.points.PointCloud(vertices=pc)
    scene.add_geometry(pc1)

    
    # print(translation.shape)
    # rand_quat2 = rand_quat.unsqueeze(1).repeat([1,1, 1])
    # translation = rot_p_by_quaterion(translation, rand_quat2)
    

   
    # quaternion = quaternion_mult(rand_quat, quaternion).squeeze().numpy()

    _transform = R.from_quat(quaternion).as_matrix()
    
    b = trimesh.primitives.Sphere(radius=0.005)
    tt = np.eye(4)
    tt[:3,:3] = _transform
    tt[:3,3] = translation
    scene.add_geometry(b, transform=tt)

    gripper = PandaGripper(root_folder='')
    gripper.apply_transformation(tt)
    scene.add_gripper(gripper)
    
    if i == 2:
        break

scene.show()