from turtle import shape
import trimesh
import numpy as np
import trimesh.transformations as tra
from pathlib import Path
from tqdm import tqdm


# define scene for debugging
class Scene(trimesh.Scene):
    """A Scene object for visualization. This object inherits trimesh.Scene class.
    All of the trimesh.Scene's functions can be used.
    Usage:
        >>> scene = Scene()
    """
    def __init__(self):
        # adds names for coordinate frames
        self.coordinate_frame_geom_names = ['center_point',
                                            'X_axis', 'X_pos_node',
                                            'Y_axis', 'Y_pos_node',
                                            'Z_axis', 'Z_pos_node']
        self.coordinate_frame_on = False

        # call parent class
        super(Scene, self).__init__()

    def add_gripper_bb(self, gripper, face_colors = [255,255,255,75]):
        '''
        Adds gripper bounding boxes to the scene.
        Arguments:
            * ``gripper``: Gripper Object.
            * ``face_colors``: RGBA for faces of bounding boxes.
        Note:
            Must be added after transformation to the gripper
        '''
        for _mesh in gripper.get_meshes():
            _mesh.bounding_box.visual.face_colors = face_colors
            self.add_geometry(_mesh.bounding_box, geom_name = _mesh.metadata['name']+'_bb')

    def add_mesh(self, mesh):
        '''
        Removes gripper's bounding box from the scene.
        Arguments:
            * ``mesh`` (instance of ``trimesh.Trimesh``): mesh to be added.
        '''
        self.add_geometry(mesh)

    def remove_gripper_bb(self, gripper):
        '''
        Removes gripper's bounding box from the scene.
        Arguments:
            * ``gripper``: Gripper Object.
        '''
        for _mesh in gripper.get_meshes():
            self.delete_geometry(_mesh.metadata['name']+'_bb')

    def add_gripper(self, gripper, face_colors = [255,255,255,150]):
        '''
        Adds gripper to the scene.
        Arguments:
            * ``gripper``: Gripper Object.
            * ``face_colors``: RGBA for faces of bounding boxes.
        '''
        for _mesh in gripper.get_meshes():
            _mesh.visual.face_colors = face_colors
            self.add_geometry(_mesh)
        
    def remove_gripper(self, gripper):
        '''
        Removes gripper from the scene.
        Arguments:
            * ``gripper``: Gripper Object.
        '''
        for _mesh in gripper.get_meshes():
            self.delete_geometry(_mesh.metadata['name'])

    def add_object(self, obj):
        '''
        Adds object to the scene. If already exists, updates it.
        Arguments:
            * ``obj``: Object.
            * ``face_colors``: RGBA for faces of object.
        '''

        # remove the object if still exists
        if obj.mesh.metadata['name'] in list(self.geometry.keys()):
            self.remove_object(obj)

        #obj.mesh.visual.face_colors = face_colors
        self.add_geometry(obj.mesh)
        
    def remove_object(self, obj):
        '''
        Removes object from the scene.
        Arguments:
            * ``obj``: Object or name of the object.
        '''
        if isinstance(obj, str):
            self.delete_geometry(obj)
        else:
            self.delete_geometry(obj.mesh.metadata['name'])

    def add_rays(self, closing_rays):
        '''
        Adds contact rays to the scene.
        Arguments:
            * ``closing_rays`` ([(3,N),(3,N)], list of 3xN ``float`` arrays): ray locations and directions.
        Note:
            Must be added after transformation to the gripper
        '''
        ray_origins, ray_directions  = closing_rays
        print(len(closing_rays))
        for i, _ray_origin in enumerate(ray_origins):
            _ray_mesh = trimesh.primitives.Sphere(radius=0.001, center=_ray_origin[:3])
            _ray_mesh.visual.face_colors = [255, 255, 0, 255]
            self.add_geometry(_ray_mesh, geom_name='ray_'+str(i))

        ray_visualize = trimesh.load_path(np.hstack((ray_origins[:,:3],
                                            #  gripper.ray_origins[:,:3] + gripper.ray_directions*0.05)).reshape(-1, 2, 3))
                                             ray_origins[:,:3] + ray_directions*0.1)).reshape(-1, 2, 3))
                                             
        self.add_geometry(ray_visualize, geom_name='ray')

    def remove_rays(self):
        '''
        Removes contact rays from the scene.
        Arguments:
            * ``gripper``: Gripper Object.
        '''
        for geom_name in list(self.geometry.keys()):
            if 'ray' in geom_name:
                self.delete_geometry(geom_name)


    def add_coordinate_frame(self):
        '''
        Adds coordinate frame along with directions to the scene.
        '''

        # do not run if coordinate frame is already visualized
        if self.coordinate_frame_on:
           return
   
        # show center point
        center_mesh = trimesh.primitives.Sphere(radius=0.001, center=(0,0,0))
        center_mesh.visual.face_colors=[0,255,255,255]
        self.add_geometry(center_mesh, geom_name=f'center_point')

        # show positive Z axis
        axis_height = 0.25
        z_tf = tra.euler_matrix(0, 0, 0)
        z_tf[:3,3] = tra.translation_matrix([0,0,axis_height/2])[:3,3]
        z_axis = trimesh.primitives.Cylinder(radius=0.0005, height=axis_height, transform=z_tf)
        z_axis.visual.face_colors=[0,0,255,255]
        self.add_geometry(z_axis, geom_name=f'Z_axis')

        # show positive Y axis
        y_tf = tra.euler_matrix(np.pi/2, 0, 0)
        y_tf[:3,3] = tra.translation_matrix([0,axis_height/2,0])[:3,3]
        y_axis = trimesh.primitives.Cylinder(radius=0.0005, height=axis_height, transform=y_tf)
        y_axis.visual.face_colors=[0,255,0,255]
        self.add_geometry(y_axis, geom_name=f'Y_axis')

        # show positive X axis
        x_tf = tra.euler_matrix(0, -np.pi/2, 0)
        x_tf[:3,3] = tra.translation_matrix([axis_height/2,0,0])[:3,3]
        x_axis = trimesh.primitives.Cylinder(radius=0.0005, height=axis_height, transform=x_tf)
        x_axis.visual.face_colors=[255,0,0,255]
        self.add_geometry(x_axis, geom_name=f'X_axis')

        _s = trimesh.primitives.Sphere(radius=0.005, center=(axis_height,0,0))
        _s.visual.face_colors=[255,0,0,255]
        self.add_geometry(_s, geom_name=f'X_pos_node')

        _s = trimesh.primitives.Sphere(radius=0.005, center=(0,axis_height,0))
        _s.visual.face_colors=[0,255,0,255]
        self.add_geometry(_s, geom_name=f'Y_pos_node')

        _s = trimesh.primitives.Sphere(radius=0.005, center=(0,0,axis_height))
        _s.visual.face_colors=[0,0,255,255]
        self.add_geometry(_s, geom_name=f'Z_pos_node')

        self.coordinate_frame_on = True

    def remove_coordinate_frame(self):
        '''
        Remove coordinate frame from the scene.
        '''
        if not self.coordinate_frame_on:
           return

        self.delete_geometry(self.coordinate_frame_geom_names)
        self.coordinate_frame_on = False
        

# define gripper
class PandaGripper(object):
    """An object representing a Franka Panda gripper."""

    def __init__(self, q=None, num_contact_points_per_finger=10, root_folder='', num_get_distance_rays=20):
        """Create a Franka Panda parallel-yaw gripper object.
        Keyword Arguments:
            q {list of int} -- configuration (default: {None})
            num_contact_points_per_finger {int} -- contact points per finger (default: {10})
            root_folder {str} -- base folder for model files (default: {''})
            face_color {list of 4 int} (optional) -- RGBA, make A less than 255 to have transparent mehs visualisation
        """
        self.joint_limits = [0.0, 0.04]
        self.default_pregrasp_configuration = 0.04
        self.num_contact_points_per_finger = num_contact_points_per_finger
        self.num_get_distance_rays = num_get_distance_rays

        if q is None:
            q = self.default_pregrasp_configuration

        self.q = q

        self.base = trimesh.load(Path(root_folder)/'assets/urdf_files/meshes/collision/hand.obj')
        self.base.metadata['name'] = 'base'
        self.finger_left = trimesh.load(Path(root_folder)/'assets/urdf_files/meshes/collision/finger.obj')
        self.finger_left.metadata['name'] = 'finger_left'
        self.finger_right = self.finger_left.copy()
        self.finger_right.metadata['name'] = 'finger_right'

        # transform fingers relative to the base
        self.finger_left.apply_transform(tra.euler_matrix(0, 0, np.pi))
        self.finger_left.apply_translation([0, -q, 0.0584])  # moves relative to y
        self.finger_right.apply_translation([0, +q, 0.0584])

        self.fingers = trimesh.util.concatenate([self.finger_left, self.finger_right])
        self.hand = trimesh.util.concatenate([self.fingers, self.base])


        # this makes to rotate the gripper to match with real world
        self.apply_transformation(tra.euler_matrix(0, 0, -np.pi/2))

        # add collision rays
        self.set_collision_rays_and_standoff()
        
        # add rays to get the distance between object and gripper hand
        self.set_get_distance_rays()

    def set_get_distance_rays(self):
        self.get_distance_ray_origins = []
        self.get_distance_ray_directions = []
        for i in np.linspace(-0.04,0.04,self.num_get_distance_rays):
            self.get_distance_ray_origins.append([i,0,0,1])
            self.get_distance_ray_directions.append([0,0,1])
        self.get_distance_ray_origins = np.asarray(self.get_distance_ray_origins)
        self.get_distance_ray_directions = np.asarray(self.get_distance_ray_directions)
        
    def set_collision_rays_and_standoff(self):
        self.ray_origins = []
        self.ray_directions = []
        for j in np.linspace(-0.004,0.004,4):
            for i in np.linspace(-0.012, 0.015, 5):
                self.ray_origins.append(
                    np.r_[self.finger_left.bounding_box.centroid + [0, j, i], 1])
                self.ray_origins.append(
                    np.r_[self.finger_right.bounding_box.centroid + [0, j, i], 1])
                # self.ray_directions.append(
                #     np.r_[+self.finger_left.bounding_box.primitive.transform[:3, 1]])
                # self.ray_directions.append(
                #     np.r_[-self.finger_right.bounding_box.primitive.transform[:3, 1]])
                self.ray_directions.append(
                    np.r_[+self.finger_left.bounding_box.primitive.transform[:3, 0]])
                self.ray_directions.append(
                    np.r_[-self.finger_right.bounding_box.primitive.transform[:3, 0]])
        self.ray_origins = np.array(self.ray_origins)
        self.ray_directions = np.array(self.ray_directions)

        self.standoff_range = np.array([max(self.finger_left.bounding_box.bounds[0, 2],
                                            self.base.bounding_box.bounds[1, 2]),
                                        self.finger_left.bounding_box.bounds[1, 2]])
        self.standoff_range[0] += 0.001

    def apply_transformation(self, transform):
        #transform = transform.dot(tra.euler_matrix(0, 0, -np.pi/2))
        # applies relative to the latest transform
        self.finger_left.apply_transform(transform)
        self.finger_right.apply_transform(transform)
        self.base.apply_transform(transform)
        self.fingers.apply_transform(transform)
        self.hand.apply_transform(transform)

        # rays:
        # self.set_collision_rays_and_standoff()


    def get_obbs(self):
        """Get list of obstacle meshes.
        Returns:
            list of trimesh -- bounding boxes used for collision checking
        """
        return [self.finger_left.bounding_box, self.finger_right.bounding_box, self.base.bounding_box]

    def get_meshes(self):
        """Get list of meshes that this gripper consists of.
        Returns:
            list of trimesh -- visual meshes
        """
        return [self.finger_left, self.finger_right, self.base]

    def get_bb(self, all=False):
        if all:
            return trimesh.util.concatenate(self.get_meshes()).bounding_box
        return trimesh.util.concatenate(self.get_meshes())

    def get_closing_rays(self, transform=np.eye(4)):
        """Get an array of rays defining the contact locations and directions on the hand.
        Arguments:
            transform {[nump.array]} -- a 4x4 homogeneous matrix
        Returns:
            numpy.array -- transformed rays (origin and direction)
        """

        return transform[:3, :].dot(
            self.ray_origins.T).T, transform[:3, :3].dot(self.ray_directions.T).T
    
    def get_transformed_get_distance_rays(self,transform=np.eye(4)):

        return transform[:3, :].dot(
            self.get_distance_ray_origins.T).T, transform[:3, :3].dot(self.get_distance_ray_directions.T).T

# define object
class Object(object):
    """Represents a graspable object."""

    def __init__(self, filename=None, name=None, scale=1.0, shift_to_center=True, mesh=None):
    # def __init__(self, filename, category):
        """Constructor.
        :param filename: Mesh to load
        :param scale: Scaling factor
        """

        self.collision_manager = trimesh.collision.CollisionManager()
        if mesh:
            self.mesh = mesh
        elif '.obj' in filename:
            self.mesh = trimesh.load(filename, force='mesh')
        else:
            self.mesh = trimesh.load(filename)
        self.scale = scale
        if self.scale != 1:
            self.rescale(self.scale)

        self.obj_mesh_mean = np.mean(self.mesh.vertices, 0)
        print("object mesh mean is :", self.obj_mesh_mean)
        if shift_to_center:            
            print("before shifting to center: ", self.mesh.vertices[0])
            self.mesh.vertices -= np.expand_dims(self.obj_mesh_mean, 0)
            print("after shifting to center: ", self.mesh.vertices[0])
            print("i am shifted to center")
            print(np.mean(self.mesh.vertices, 0))
        self.mesh.metadata['name'] = name

        self.filename = filename
        if isinstance(self.mesh, list):
            # this is fixed in a newer trimesh version:
            # https://github.com/mikedh/trimesh/issues/69
            print("Warning: Will do a concatenation")
            self.mesh = trimesh.util.concatenate(self.mesh)
       
        self.reset_collision_manager()

    def get_obj_mesh_mean(self):
        return np.array(self.obj_mesh_mean)

    def reset_collision_manager(self):
        try:
            self.collision_manager.remove_object('object')
        except:
            pass
        self.collision_manager.add_object('object', self.mesh)


    def apply_transform(self, transform):
        """Apply transform for the object mesh.
        :param transform
        """
        self.mesh.apply_transform(transform)
        self.collision_manager.set_transform('object', transform)

    def rescale(self, scale=1.0):
        """Set scale of object mesh.
        :param scale
        """
        self.scale = scale
        #print("scaling with: ",self.scale)
        self.mesh.apply_scale(self.scale)
        self.reset_collision_manager()


    def in_collision_with(self, mesh, transform=np.eye(4)):
        """Check whether the object is in collision with the provided mesh.
        :param mesh:
        :param transform:
        :return: boolean value
        """
        return self.collision_manager.in_collision_single(mesh, transform=transform)

    def min_distance_single(self, mesh, transform=np.eye(4)):
        return self.collision_manager.in_collision_single(mesh, transform=transform)
