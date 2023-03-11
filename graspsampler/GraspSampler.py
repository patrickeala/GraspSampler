from .common import PandaGripper, Scene, Object
from .PointCloudManager import PointCloudManager
from .utils import sample_multiple_grasps, perturb_transform, perturb_grasp_locally, sample_multiple_grasps_bowl, sample_multiple_grasps_mug, sample_equal_spaced_grasps,get_minimum_gripper_distances_to_object
import trimesh
import trimesh.transformations as tra
import numpy as np
import pickle
import copy

class GraspSampler(object):
    def __init__(self, root_dir='', gripper=None, seed=40):
        
        self.seed = seed
        np.random.seed(self.seed)

        if gripper == None:
            self.gripper = PandaGripper(root_folder=root_dir)

        self.scene = Scene() # used for visualization
        self.scene.add_gripper(self.gripper)

        self.obj = None
        self.obj_pc = None
        self.trans_pose = None
        self.pc_manager = PointCloudManager(seed=self.seed)


    def save(self, obj_name):
        filename = f"sampled_grasps/{obj_name}_grasps.pkl"
        file_to_write = open(filename, "wb")
        pickle.dump(self.sampled_data, file_to_write)
        print(f"Saved {filename}")

    def update_object(self, obj_filename=None, name=None, obj_scale=1.0, mesh=None):
        # adds objects, if exists, removes and reuploads
        self.obj = Object(filename=obj_filename, scale=obj_scale, name=name, mesh=mesh)
        #self.scene.add_object(self.obj, face_colors=[0, 255, 0, 150])
  
    def get_pc(self, obj_pose=np.eye(4), depth_noise=0.0, full_pc=False):
        '''
        Render pc for object with location at ```obj_pose```.
        Arguments:
            * ``obj_pose`` ((4,4), ``float``): homogenous transformation of the object relative to the world frame.
            * ``depth_noise`` (``float``): depth noise, default=0.005.
        Returns:
            * pc((3,N), ``float``): Point cloud of the rendered scene.
        '''
        self.pc_manager.update_object(obj_mesh=self.obj.mesh)
        self.pc_manager.set_obj_pose(np.eye(4))
        _, _, pc, transferred_pose, camera_pose = self.pc_manager.render_pc(depth_noise=depth_noise, full_pc=full_pc)
        
        return pc, transferred_pose, camera_pose

    def view_pc(self, pc):
        self.pc_manager.view_pointcloud(pc)


    def get_multiple_random_pcs(self, number_of_pcs, target_pc_size=None, depth_noise=0.005, dropout=0.001):
        '''
        Render multiple random ``number_of_pcs`` pcs for given object.
        Arguments:
            * ``number_of_pcs`` (``int``): number of pcs to be rendered.
            * ``depth_noise`` (``float``): depth noise, default=0.005.
        Returns:
            * pcs (list of ((3,N), ) ``float``): Point cloud of the rendered scene.
        '''
        self.pc_manager.update_object(obj_mesh=self.obj.mesh)
        self.pc_manager.set_obj_pose(np.eye(4))
        _, depths, pcs, transfered_poses, camera_poses = self.pc_manager.render_multiple_pcs(number_of_pcs=number_of_pcs, target_pc_size=target_pc_size, depth_noise=depth_noise, dropout=dropout)
        return pcs, depths, transfered_poses, camera_poses


    def sample_grasps(self, number_of_grasps=10, 
                            alpha_lim = [0, 2*np.pi],
                            beta_lim = [0, 0],
                            gamma_lim = [0, 0],
                            silent=False):
        '''
        Sample random ``number_of_grasps`` grasps.
        Arguments:
            * ``number_of_grasps`` (``int``): number of grasps to be sampled.
            * ``silent`` (``bool``): verbosity, default=False.
            * ``quality_type`` (``string``): type of grasp quality Default=``weighted_antipodal``.
        Returns:
            * ``points`` ((n_perturbed_grasps,3) ``float``): points on the trimesh face for each grasp 
            * ``normals`` ((n_perturbed_grasps,3) ``float``): normal vectors on the trimesh face from points
            * ``transforms`` ((n_perturbed_grasps,4,4) ``float``): perturbed homogenous transform matrices for the gripper
            * ``roll_angles`` ((n_perturbed_grasps,) ``float``): roll angles for each grasp around the normal (right hand rule) 
            * ``standoffs`` ((n_perturbed_grasps,) ``float``): offset for each perturbed grasp from the points on the trimesh face 
            * ``collisions`` ((n_perturbed_grasps,), ``float``) : collision points of each grasp with the object 
            * ``qualities`` ((n_perturbed_grasps,) ``float``): quality score of grasps
            * ``quality_types`` ((n_perturbed_grasps,) ``string``): type of quality metric used to evaluate grasps
        '''
        return sample_multiple_grasps(
                self.obj.mesh, self.gripper,
                number_of_grasps=number_of_grasps,
                alpha_lim = alpha_lim,
                beta_lim = beta_lim,
                gamma_lim = gamma_lim,
                silent=silent)

    def sample_grasps_bowl(self, number_of_grasps=10, 
                            alpha_lim = [0, 2*np.pi],
                            beta_lim = [0, 0],
                            gamma_lim = [0, 0],
                            silent=False):
        '''
        Sample random ``number_of_grasps`` grasps.
        Arguments:
            * ``number_of_grasps`` (``int``): number of grasps to be sampled.
            * ``silent`` (``bool``): verbosity, default=False.
            * ``quality_type`` (``string``): type of grasp quality Default=``weighted_antipodal``.
        Returns:
            * ``points`` ((n_perturbed_grasps,3) ``float``): points on the trimesh face for each grasp 
            * ``normals`` ((n_perturbed_grasps,3) ``float``): normal vectors on the trimesh face from points
            * ``transforms`` ((n_perturbed_grasps,4,4) ``float``): perturbed homogenous transform matrices for the gripper
            * ``roll_angles`` ((n_perturbed_grasps,) ``float``): roll angles for each grasp around the normal (right hand rule) 
            * ``standoffs`` ((n_perturbed_grasps,) ``float``): offset for each perturbed grasp from the points on the trimesh face 
            * ``collisions`` ((n_perturbed_grasps,), ``float``) : collision points of each grasp with the object 
            * ``qualities`` ((n_perturbed_grasps,) ``float``): quality score of grasps
            * ``quality_types`` ((n_perturbed_grasps,) ``string``): type of quality metric used to evaluate grasps
        '''
        return sample_multiple_grasps_bowl(
                self.obj.mesh, self.gripper,
                number_of_grasps=number_of_grasps,
                alpha_lim = alpha_lim,
                beta_lim = beta_lim,
                gamma_lim = gamma_lim,
                silent=silent)

    def sample_grasps_mug(self, number_of_grasps=10, 
                            alpha_lim = [0, 2*np.pi],
                            beta_lim = [0, 0],
                            gamma_lim = [0, 0],
                            silent=False,
                            jcups=False):
        '''
        Sample random ``number_of_grasps`` grasps.
        Arguments:
            * ``number_of_grasps`` (``int``): number of grasps to be sampled.
            * ``silent`` (``bool``): verbosity, default=False.
            * ``quality_type`` (``string``): type of grasp quality Default=``weighted_antipodal``.
        Returns:
            * ``points`` ((n_perturbed_grasps,3) ``float``): points on the trimesh face for each grasp 
            * ``normals`` ((n_perturbed_grasps,3) ``float``): normal vectors on the trimesh face from points
            * ``transforms`` ((n_perturbed_grasps,4,4) ``float``): perturbed homogenous transform matrices for the gripper
            * ``roll_angles`` ((n_perturbed_grasps,) ``float``): roll angles for each grasp around the normal (right hand rule) 
            * ``standoffs`` ((n_perturbed_grasps,) ``float``): offset for each perturbed grasp from the points on the trimesh face 
            * ``collisions`` ((n_perturbed_grasps,), ``float``) : collision points of each grasp with the object 
            * ``qualities`` ((n_perturbed_grasps,) ``float``): quality score of grasps
            * ``quality_types`` ((n_perturbed_grasps,) ``string``): type of quality metric used to evaluate grasps
        '''
        return sample_multiple_grasps_mug(
                self.obj.mesh, self.gripper,
                number_of_grasps=number_of_grasps,
                alpha_lim = alpha_lim,
                beta_lim = beta_lim,
                gamma_lim = gamma_lim,
                silent=silent,
                jcups=jcups)

    def sample_equal_spaced_grasps(self, extend = 0.35, step = 0.025, quats_per_point = 5, silent=False):
        '''
        Equal spaced grasps. Count of grasps depend on parameters.
        '''
        return sample_equal_spaced_grasps(
                self.obj.mesh, self.gripper,
                extend=extend, step=step, quats_per_point=quats_per_point,
                silent=silent)

    def perturb_grasp_locally(self, number_of_grasps,
                              normals, origins, alpha, beta, gamma,
                              alpha_range=np.pi/36,
                              beta_range=np.pi/36,
                              gamma_range=np.pi/36,
                              t_range = 0.005,
                              silent=False):

        return perturb_grasp_locally(self.obj.mesh, self.gripper,
                                     number_of_grasps=number_of_grasps,
                                     normals=normals,
                                     origins=origins,
                                     alpha=alpha,
                                     beta=beta,
                                     gamma=gamma,
                                     alpha_range=alpha_range,
                                     beta_range=beta_range,
                                     gamma_range=gamma_range,
                                     t_range = t_range,
                                     silent=silent)


    def perturb_transform(self, transform, number_of_grasps, 
                      min_translation=(-0.03,-0.03,-0.03),
                      max_translation=(0.03,0.03,0.03),
                      min_rotation=(-0.6,-0.2,-0.6),
                      max_rotation=(+0.6,+0.2,+0.6)):
        '''
        Perturbs given transform within translation_range and rotation_range
        Arguments:
            * ``transform`` ((4,4) ``float``): homogenous transform matrix.
            * ``number_of_grasps`` ((3,) ``float``): number of perturbed grasps.
            * ``translation_range`` (``float``): perturbation range for the translation, in meters (this defines perturbation)..
            * ``rotation_range`` (``float``): perturbation range for the rotation, in meters (this defines perturbation)..
        Returns:
            * ``transforms`` (list of (4,4) ``float`` matrices): perturbed grasps.
        '''
        return perturb_transform(transform, number_of_grasps, 
                                min_translation=min_translation,
                                max_translation=max_translation,
                                min_rotation=min_rotation,
                                max_rotation=max_rotation)

    def get_minimum_distance_to_obj(self,transforms):
        return get_minimum_gripper_distances_to_object(transforms,self.obj.mesh,self.gripper)

    def grasp_visualize(self, transform,
                        coordinate_frame=False,
                        grasp_debug_items=False,
                        other_debug_items=False,
                        point=None,
                        origin=None):
        # step 1: remove object from the scene
        # self.scene.remove_object(self.obj)
        self.scene.remove_gripper(self.gripper)
        

        # step 2: apply transform to the gripper
        _gripper = copy.deepcopy(self.gripper)
        _gripper.apply_transformation(transform=transform)
        self.scene.add_gripper(_gripper)


        # # step 3: apply transform to the gripper
        # _gripper2 = copy.deepcopy(self.gripper)
        # _gripper2.apply_transformation(transform=transform2)
        # for zz, _mesh in enumerate(_gripper2.get_meshes()):
        #     _mesh.visual.face_colors = [255,255,0,150]
        #     self.scene.add_geometry(_mesh, f'gp{zz}')
        

        # step 3: add object to the scene
        #self.scene.add_object(self.obj, face_colors=[0, 255, 0, 170])
        self.scene.add_object(self.obj)

        # step 4: add debug tools
        if coordinate_frame:
            self.scene.add_coordinate_frame()

        if grasp_debug_items:
            self.scene.add_rays(_gripper.get_closing_rays(transform))
            # self.scene.add_rays(_gripper.get_transformed_get_distance_rays(transform))

        if other_debug_items:
            if point is not None:
                point_sphere = trimesh.primitives.Sphere(center=point, radius=0.001)
                point_sphere.visual.face_colors = [255,0,0,255]
                self.scene.add_geometry(point_sphere, 'point')
            if origin is not None:
                point_sphere = trimesh.primitives.Sphere(center=origin, radius=0.001)
                point_sphere.visual.face_colors = [255,255,0,255]
                self.scene.add_geometry(point_sphere, 'point2')

        # visualize
        self.scene.show()

        # clear the scene
        self.scene.remove_rays()
        self.scene.remove_coordinate_frame()
        self.scene.delete_geometry('point')
        self.scene.delete_geometry('point2')
        # self.scene.delete_geometry('gp0')
        # self.scene.delete_geometry('gp1')
        # self.scene.delete_geometry('gp2')


        # restart the gripper
        self.scene.remove_gripper(self.gripper)
        self.scene.add_gripper(self.gripper)
