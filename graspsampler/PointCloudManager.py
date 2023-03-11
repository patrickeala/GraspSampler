from numpy.core.fromnumeric import std
import pyrender
import trimesh
import trimesh.transformations as tra
import cv2
import numpy as np
from .utils import trans_matrix, regularize_pc_point_count
from scipy.interpolate import griddata
from scipy import ndimage

def add_gaussian_shifts(depth, std=1/2.0):

    rows, cols = depth.shape 
    gaussian_shifts = np.random.normal(0, std, size=(rows, cols, 2))
    gaussian_shifts = gaussian_shifts.astype(np.float32)

    # creating evenly spaced coordinates  
    xx = np.linspace(0, cols-1, cols)
    yy = np.linspace(0, rows-1, rows)

    # get xpixels and ypixels 
    xp, yp = np.meshgrid(xx, yy)

    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    xp_interp = np.minimum(np.maximum(xp + gaussian_shifts[:, :, 0], 0.0), cols)
    yp_interp = np.minimum(np.maximum(yp + gaussian_shifts[:, :, 1], 0.0), rows)

    depth_interp = cv2.remap(depth, xp_interp, yp_interp, cv2.INTER_LINEAR)

    return depth_interp
    

def filterDisp(disp, dot_pattern_, invalid_disp_):

    size_filt_ = 9

    xx = np.linspace(0, size_filt_-1, size_filt_)
    yy = np.linspace(0, size_filt_-1, size_filt_)

    xf, yf = np.meshgrid(xx, yy)

    xf = xf - int(size_filt_ / 2.0)
    yf = yf - int(size_filt_ / 2.0)

    sqr_radius = (xf**2 + yf**2)
    vals = sqr_radius * 1.2**2 

    vals[vals==0] = 1 
    weights_ = 1 /vals  

    fill_weights = 1 / ( 1 + sqr_radius)
    fill_weights[sqr_radius > 9] = -1.0 

    disp_rows, disp_cols = disp.shape 
    dot_pattern_rows, dot_pattern_cols = dot_pattern_.shape

    lim_rows = np.minimum(disp_rows - size_filt_, dot_pattern_rows - size_filt_)
    lim_cols = np.minimum(disp_cols - size_filt_, dot_pattern_cols - size_filt_)

    center = int(size_filt_ / 2.0)

    window_inlier_distance_ = 0.1

    out_disp = np.ones_like(disp) * invalid_disp_

    interpolation_map = np.zeros_like(disp)

    for r in range(0, lim_rows):

        for c in range(0, lim_cols):

            if dot_pattern_[r+center, c+center] > 0:
                                
                # c and r are the top left corner 
                window  = disp[r:r+size_filt_, c:c+size_filt_] 
                dot_win = dot_pattern_[r:r+size_filt_, c:c+size_filt_] 
  
                valid_dots = dot_win[window < invalid_disp_]

                n_valids = np.sum(valid_dots) / 255.0 
                n_thresh = np.sum(dot_win) / 255.0 

                if n_valids > n_thresh / 1.2: 

                    mean = np.mean(window[window < invalid_disp_])

                    diffs = np.abs(window - mean)
                    diffs = np.multiply(diffs, weights_)

                    cur_valid_dots = np.multiply(np.where(window<invalid_disp_, dot_win, 0), 
                                                 np.where(diffs < window_inlier_distance_, 1, 0))

                    n_valids = np.sum(cur_valid_dots) / 255.0

                    if n_valids > n_thresh / 1.2: 
                    
                        accu = window[center, center] 

                        assert(accu < invalid_disp_)

                        out_disp[r+center, c + center] = round((accu)*8.0) / 8.0

                        interpolation_window = interpolation_map[r:r+size_filt_, c:c+size_filt_]
                        disp_data_window     = out_disp[r:r+size_filt_, c:c+size_filt_]

                        substitutes = np.where(interpolation_window < fill_weights, 1, 0)
                        interpolation_window[substitutes==1] = fill_weights[substitutes ==1 ]

                        disp_data_window[substitutes==1] = out_disp[r+center, c+center]

    return out_disp


class PointCloudManager(object):
    """An object to manage point cloud generation. Uses pyrender in a core functions"""
    def __init__(self, obj_mesh=None,
                       camera=None,
                       nodes=None,
                       bg_color=None,
                       ambient_light=None,
                       name=None,
                       width=400,
                       height=400,
                       point_size=1.0,
                       fit_coefficient=5.0,
                       seed=140):

        '''
        PointCloudManager costructor.
        Arguments:
            * ``obj_mesh`` (instance of ``trimesh.Trimesh``): mesh of the object to be added to pyrender scene.
            * ``camera`` (instance of ``pyrender.camera.Camera``): Camera for the pyrender scene. Default pyrender default camera.
            * ``bg_color`` ((4,) ``float``, optional): Background color of scene.
            * ``ambient_light`` ((3,) ``float``, optional): Color of ambient light. Defaults to no ambient light..
            * ``name`` (``str``, optional): The user-defined name of this object.
            * ``width`` ((``int``, optional): Viewer width, default 400.
            * ``height`` (``int``, optional): Viewer height, default 400.
            * ``point_size`` (``float``, optional): The pixel size of points in point clouds. Default 1.0.
            * ``fit_coefficient`` (``float``, optional): Defines how far  the object located from world frame.
            If set >= 1.0, fully visible. Default 5.0.
        '''

        # set seed
        np.random.seed(seed)

        # init pyrender scene
        self.scene = pyrender.Scene(nodes=nodes, bg_color=bg_color, ambient_light=ambient_light, name=name)
        
        # init camera
        self.camera_node = self.set_camera(camera=camera)

        # global variable
        self.fit_coefficient = fit_coefficient

        # upload object
        self.obj_node = None
        if obj_mesh is not None:    
            self.update_object(obj_mesh)
            

        # init offscreen reenderer: do not change
        self.width = width
        self.height = height
        self.point_size = point_size

        # generate random poses
        self.random_poses = self._random_poses_generator()

    def _random_poses_generator(self):
        random_poses = []
        for az in np.linspace(0, np.pi * 2, 360):
            for el in np.linspace(-np.pi / 2, np.pi / 2, 360):
                random_poses.append(tra.euler_matrix(el, az, 0))
        return random_poses

    def dropout_pointcloud(self, pointcloud, dropout_ratio=0.875):
        drop_idx = np.where(np.random.random((pointcloud.shape[0]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            pointcloud[drop_idx,:] = pointcloud[0,:] # set to the first point
        return pointcloud

    def update_object(self, obj_mesh):
        '''
        Updates object in the scene. If object is not present, creates new one,
        otherwise replaces old one.
        Arguments:
            * ``obj_mesh`` (instance of ``trimesh.Trimesh``): mesh of the object to be added to pyrender scene.
        Returns:
            * obj_node (instance of ``pyrender.Node): object node created within graph.
        '''
        if self.obj_node is not None:
            self.scene.remove_node(self.obj_node)

        # shift to the center and get distance from camera
        lbs = np.min(obj_mesh.vertices, 0)
        ubs = np.max(obj_mesh.vertices, 0)
        self.obj_distance = np.max(ubs - lbs) * self.fit_coefficient

        self.mesh = pyrender.Mesh.from_trimesh(obj_mesh)
        self.obj_node = self.add_to_scene(self.mesh, name='object')
        return self.obj_node

    def set_obj_pose(self, obj_pose, adjust_object_to_fit=True):
        '''
        sets specified pose to the object in the scene
        Arguments:
            * ``obj_pose`` ((4,4) ``float``): Homogenous transformation matrix of the object relative to world frame.
        Returns:
            * obj_pose ((4,4) ``float``): Final pose of the object relative to the world.
        '''

        # check object node
        if self.obj_node is None:
            raise ValueError('No object in the scene!')

        transferred_pose = obj_pose.copy()
        if adjust_object_to_fit:
            transferred_pose[2, 3] = self.obj_distance
        self.scene.set_pose(self.obj_node, transferred_pose)

        return transferred_pose

    def reset_object_pose(self):
        pose = np.eye(4)
        # pose[0, 0] = 0
        # pose[0, 1] = 1
        # pose[1, 0] = -1
        # pose[1, 1] = 0
        self.set_obj_pose(pose, False)

    def render_pc(self, full_pc=False, depth_noise=[1,1], dropout=0.01):
        '''
        Renders pointcloud for the scene. If ``full_pc`` is ``True``, 
        then pointcloud is generated for all faces (vertices) and requires no rendering.
        Arguments:
            * ``full_pc`` (``Boolean``, optional): Set this parameter to true if need to get full point cloud of the object.
            * ``depth_noise`` [param1,param2]: Adds noise to the depth rendering, given in percentage.
        Returns:
            * color ((W,H) ``int``): RGB image of the rendered scene.
            * depth ((W,H) ``depth``): Depth image of the rendered scene.
            * pc ((3,N) ``float``): Point cloud of the rendered scene.
        '''
        # check object node
        if self.obj_node is None:
            raise ValueError('No object in the scene!')
        self.remove_camera()

        self.camera_node=self.set_camera()
        print("camera_pose: ",self.camera_pose)
        # # set distance from object to camera to be fittable to the scene
        obj_pose = self.scene.get_pose(self.obj_node)
        #print('From render_pc:')
        #print(obj_pose)
        transferred_pose = obj_pose.copy()
        print("transfered obj pose： ",transferred_pose)
        #transferred_pose[2, 3] = self.obj_distance
        #self.set_obj_pose(transferred_pose)

        # return perfect point cloud (all covered)
        if full_pc:
            # https://github.com/mmatl/pyrender/issues/14
            points = self.scene.get_nodes(node=self.obj_node)
            points = next(iter(points))
            points = points.mesh.primitives[0].positions
            # TODO: why not full for box and cylinder?:
            # may not produce full point cloud for perfect triangular meshes
            return None, None, points, transferred_pose, self.camera_pose
        print("obj_pose",obj_pose)
     
    
        self.renderer = pyrender.OffscreenRenderer(viewport_width=self.width, 
                                                   viewport_height=self.height, 
                                                   point_size=self.point_size)
        color, depth = self.renderer.render(self.scene)
        self.renderer.delete()
        
#         depth = cv2.resize(depth, (400, 400))
# # cv2.imshow("image", image)
#         cv2.imshow('Noise',depth)
#         cv2.waitKey(1000)
        # print(self.scene.cameras.translation)
        # self.remove_camera()
        # self.camera_node=self.set_camera(z=obj_pose[2,3]+0.5)
        # print("set camera")
       
        # depth noise
        if depth_noise != [0,0]:
            noise_depth = self.add_noise_to_depth(depth)
            pc = self.depth_to_pointcloud(noise_depth)
        else:
            pc = self.depth_to_pointcloud(depth)
       
        if dropout > 0:
            pc = self.dropout_pointcloud(pc,dropout_ratio=dropout)
        # dropout noise
   
        # self.reset_object_pose()
        return color, depth, pc, transferred_pose, self.camera_pose


    def render_multiple_pcs(self, number_of_pcs, target_pc_size=None, depth_noise=0.005, dropout=0.001, get_color_depth=True):
        colors, depths, pcs, transferred_poses, camera_poses  = [],[],[],[],[]
        #transferred_poses = np.empty([number_of_pcs,4,4])
        # print(self.fov)
        if target_pc_size is not None:
            pcs = np.empty([number_of_pcs,target_pc_size,3])

        for i in range(number_of_pcs):
        
            # #step 1: get random poses
            # obj_pose_random = self.random_poses[np.random.randint(0, high=len(self.random_poses))]

            # #step 2: make sure it fits well to the screen
            # _camera_distance_random = 2*np.random.random()+self.obj_distance
            # obj_pose_random[2, 3] = _camera_distance_random
            # #print(_camera_distance_random)
            # #print('From render_multiple_pcs:')
            # #print(obj_pose_random)
            # self.set_obj_pose(obj_pose_random, adjust_object_to_fit=False)
            # #transferred_poses.append(obj_pose_random)
            #step 3: sample pcs
            self.reset_object_pose()
            color, depth, pc, transferred_pose, camera_pose = self.render_pc(depth_noise=depth_noise, dropout=dropout)
            print("transfer pose",transferred_pose)
            # step 4: bring the point clouds to original locations
            camera_poses.append(camera_pose)
            if target_pc_size is not None:
                pc = regularize_pc_point_count(pc, target_pc_size, True)
                new_pc = trimesh.points.PointCloud(pc)
                # new_pc.apply_transform(np.linalg.inv(camera_pose))
                new_pc = new_pc.vertices.view(np.ndarray)
                pcs[i] = new_pc
            else:
                new_pc = trimesh.points.PointCloud(pc)
                print("*(********************")
                # new_pc.apply_transform(np.linalg.inv(camera_pose))
                new_pc = new_pc.vertices.view(np.ndarray)
            pcs.append(new_pc)

            if get_color_depth:
                colors.append(color)
                depths.append(depth)
                transferred_poses.append(transferred_pose)
            #transferred_poses[i] = transferred_pose
        return colors, depths, pcs, transferred_poses, camera_poses#, transferred_poses


    def remove_object(self):
        '''
        Removes object from the scene.
        '''
        if self.obj_node is not None:
            self.scene.remove_node(self.obj_node)
            self.mesh = None
            self.obj_node = None

    def clear_scene(self):
        '''
        Clears the scene including object and camera.
        '''
        for node in self.scene.get_nodes():
            if self.scene.has_node(node):
                self.scene.remove_node(node)
        self.obj_node = None

    def view_pointcloud(self, pc):
        '''
        Visualizes point cloud.
        Arguments:
            * ``pc`` ((3,N) ``float``): Pointcloud.
        Note:
            This function hangs!
        '''
        pc_mesh = trimesh.points.PointCloud(pc)
        pc_mesh.show()

    def view_scene(self, lighting=True):
        '''
        Visualizes current scene.
        Arguments:
            * ``pc`` ((3,N) ``float``): Pointcloud.
        '''
        pyrender.Viewer(self.scene, use_raymond_lighting=lighting)

    def build_matrix_of_indices(self, height, width):
        """Build a [height, width, 2] numpy array containing coordinates.
        Args:
            height: int.
            width: int.
        Returns:
            np.ndarray B [H, W, 2] s.t. B[..., 0] contains y-coordinates, B[..., 1] contains x-coordinates
        """
        return np.indices((height, width), dtype=np.float32).transpose(1,2,0)

    def compute_xyz(self, depth_img, camera_params):
        """Compute ordered point cloud from depth image and camera parameters.
            
        Assumes camera uses left-handed coordinate system, with 
            x-axis pointing right
            y-axis pointing up
            z-axis pointing "forward"
        Args:
            depth_img: a [H, W] numpy array of depth values in meters
            camera_params: a dictionary with camera parameters
        Returns:
            a [H, W, 3] numpy array
        """
        fy = fx = 0.5 / np.tan(self.fov * 0.5) # aspectRatio is one.

        if 'x_offset' in camera_params and 'y_offset' in camera_params:
            x_offset = camera_params['x_offset']
            y_offset = camera_params['y_offset']
        else: # simulated data
            x_offset = camera_params['img_width']/2
            y_offset = camera_params['img_height']/2

        indices = self.build_matrix_of_indices(camera_params['img_height'], camera_params['img_width'])
        indices[..., 0] = np.flipud(indices[..., 0])  # pixel indices start at top-left corner. for these equations, it starts at bottom-left
        z_e = depth_img
        x_e = (indices[..., 1] - x_offset) * z_e / fx
        y_e = (indices[..., 0] - y_offset) * z_e / fy
        xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # [H, W, 3]

        return xyz_img

    def depth_to_pointcloud(self, depth):
        '''
        Converts depth to pointcloud.
        Arguments:
            * ``depth`` ((W,H) ``float``): Depth data.
        Returns:
            * pc ((3,N) ``float``): Pointcloud.
W        '''
        fy = fx = 0.5 / np.tan(self.fov * 0.5) # aspectRatio is one.


        height = depth.shape[0]
        width = depth.shape[1]

        mask = np.where(depth > 0)
        
        x = mask[1]
        y = mask[0]
        
        normalized_x = (x.astype(np.float32) - width * 0.5) / width
        normalized_y = (y.astype(np.float32) - height * 0.5) / height
        
        world_x = normalized_x * depth[y, x] / fx
        world_y = normalized_y * depth[y, x] / fy
        world_z = depth[y, x]
        ones = np.ones(world_z.shape[0], dtype=np.float32)

        return np.vstack((world_x, world_y, world_z)).T

    def add_to_scene(self, obj, pose=np.eye(4), name=None, parent_node=None, parent_name=None):
        '''
        Adds object such as camera and trimesh object to the scene.
        Arguments:
            * ``obj`` (instance of ``trimesh.Trimesh`` or ``pyrender.camera.Camera``): Object to be added.
            * ``pose`` ((4,4) ``float``, optional): Pose of the object.
            * ``name`` (``str``, optional): Name of the object.
            * ``parent_node`` (instance of ``pyrender.Node``, optional): Parent node in the graph.
            * ``parent_name`` (``str``, optional): Name of the parent node.
        Returns:
            * object_node (instance of ``pyrender.Node``): Node of the object in the graph.
        '''
        return self.scene.add(obj=obj, pose=pose, name=None, parent_node=parent_node, parent_name=parent_name)

    def remove_camera(self):
        self.scene.remove_node(self.camera_node)
        self.camera_node = None

        return

    def set_camera(self, camera=None,z = 1.2):
        '''
        Sets camera in the scene.
        Arguments:
            * ``camera`` (instance of ``pyrender.camera.Camera``): Camera for the pyrender scene. Default pyrender default camera.
        Returns:
            * camera_node (instance of ``pyrender.Node``): Node of the camera object in the graph.
        '''

        if camera is None: # default camera settings
            self.fov = np.pi/6.0
            self.aspect_ratio = 1.0
            self.znear = 0.001
            self.camera = pyrender.PerspectiveCamera(yfov=self.fov, 
                                                     aspectRatio=self.aspect_ratio, 
                                                     znear=self.znear)
            # self.camera_pose = trans_matrix(euler=[np.pi,0,0], translation=[0.0,0.0,0])
            z = np.random.uniform(0.3,0.7)
            x = np.random.uniform(-0.6,0.6)
            y = np.random.uniform(-0.6,0.6)
            translation =  np.array([x,y,z])
            
            # y = 0
            '''
            print(translation)
            beta = np.arctan2(x, translation[2])
            alpha = -np.arctan2(y, translation[2])
            euler = [alpha,beta,0]
            self.camera_pose = trans_matrix(euler=euler, translation=translation)
            '''
            orientation = -translation
            orientation = orientation / np.linalg.norm(orientation)
            s2 = np.dot(tra.translation_matrix(translation), trimesh.geometry.align_vectors([0, 0, -1],orientation))
            angle_x = tra.quaternion_matrix(
             tra.quaternion_about_axis(0, [0, 0, 1]))
            angle_y = tra.quaternion_matrix(
             tra.quaternion_about_axis(0, [0, 1, 0]))
            angle_z = tra.quaternion_matrix(
             tra.quaternion_about_axis(0, [1, 0, 0]))
            _transform = np.dot( np.dot(np.dot( s2, angle_x),angle_y),angle_z)
            # print(_transform)
            _transform[:3,3] = translation
            transforms = _transform
            self.camera_pose = transforms

            #self.camera_pose = np.eye(4)
        else:
            # TODO: add true/real cameras such as Intel Realsense D435
            raise NotImplementedError
            
        return self.add_to_scene(self.camera, pose=self.camera_pose, name='camera') # do not change this
        

    def add_coordinate_frame(self):
        '''
        Adds coordinate frame to the scene. Use it only for visualization purposes.
        '''
        ## show center point
        center_mesh = trimesh.primitives.Sphere(radius=0.001, center=(0,0,0))
        center_mesh.visual.face_colors=[0,255,255,255]
        self.add_to_scene(pyrender.Mesh.from_trimesh(center_mesh, smooth=False))

        axis_height = 0.25
        z_tf = tra.euler_matrix(0, 0, 0)
        z_tf[:3,3] = tra.translation_matrix([0,0,axis_height/2])[:3,3]
        z_axis = trimesh.primitives.Cylinder(radius=0.0005, height=axis_height, transform=z_tf)
        z_axis.visual.face_colors=[0,0,255,255]
        self.add_to_scene(pyrender.Mesh.from_trimesh(z_axis, smooth=False))

        y_tf = tra.euler_matrix(np.pi/2, 0, 0)
        y_tf[:3,3] = tra.translation_matrix([0,axis_height/2,0])[:3,3]
        y_axis = trimesh.primitives.Cylinder(radius=0.0005, height=axis_height, transform=y_tf)
        y_axis.visual.face_colors=[0,255,0,255]
        self.add_to_scene(pyrender.Mesh.from_trimesh(y_axis, smooth=False))

        x_tf = tra.euler_matrix(0, -np.pi/2, 0)
        x_tf[:3,3] = tra.translation_matrix([axis_height/2,0,0])[:3,3]
        x_axis = trimesh.primitives.Cylinder(radius=0.0005, height=axis_height, transform=x_tf)
        x_axis.visual.face_colors=[255,0,0,255]
        self.add_to_scene(pyrender.Mesh.from_trimesh(x_axis, smooth=False))


        _s = trimesh.primitives.Sphere(radius=0.005, center=(axis_height,0,0))
        _s.visual.face_colors=[255,0,0,255]
        self.add_to_scene(pyrender.Mesh.from_trimesh(_s, smooth=False))

        _s = trimesh.primitives.Sphere(radius=0.005, center=(0,axis_height,0))
        _s.visual.face_colors=[0,255,0,255]
        self.add_to_scene(pyrender.Mesh.from_trimesh(_s, smooth=False))

        _s = trimesh.primitives.Sphere(radius=0.005, center=(0,0,axis_height))
        _s.visual.face_colors=[0,0,255,255]
        self.add_to_scene(pyrender.Mesh.from_trimesh(_s, smooth=False))

    def add_noise_to_depth(self, depth):
        dot_pattern_ = cv2.imread("graspsampler/kinect-pattern_3x3.png", 0)
        # print(dot_pattern_)


        # various variables to handle the noise modelling
        scale_factor  = 200  # converting depth from m to cm 
        focal_length  = 0.5 / np.tan(self.fov * 0.5)   # focal length of the camera used 
        baseline_m    = 100 # baseline in m 
        invalid_disp_ = 99999999.9

        depth_uint16 = depth
        h, w = depth_uint16.shape 

        # Our depth images were scaled by 5000 to store in png format so dividing to get 
        # depth in meters 
        depth = depth_uint16.astype('float') 
        
        std = np.random.uniform(0.1,0.5)

        depth_interp = add_gaussian_shifts(depth,std=std)

        disp_= focal_length * baseline_m / (depth_interp + 1e-10)

        depth_f = np.round(disp_ * 8.0)/8.0

        out_disp = filterDisp(depth_f, dot_pattern_, invalid_disp_)

        depth = focal_length * baseline_m / out_disp
        depth[out_disp == 99999999.9] = 0 
        

        # # The depth here needs to converted to cms so scale factor is introduced 
        # # though often this can be tuned from [100, 200] to get the desired banding / quantisation effects 
        noisy_depth = (35130/np.round((35130/np.round(depth*scale_factor)) + np.random.normal(size=(h, w))*(1.0/6.0) + 0.5))/scale_factor 

   
        # noisy_depth = noisy_depth.astype('uint16')
        noisy_depth = depth
        # Displaying side by side the orignal depth map and the noisy depth map with barron noise cvpr 2013 model
        # cv2.namedWindow('Adding Kinect Noise', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('Adding Kinect Noise', np.hstack((depth_uint16, noisy_depth)))
        # cv2.imshow('Adding Kinect Noise', np.hstack((depth_uint16, noisy_depth)))
        # key = cv2.waitKey(1)

        return noisy_depth