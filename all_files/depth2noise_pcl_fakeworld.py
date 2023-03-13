from matplotlib.transforms import Transform
import cv2
import numpy as np


fov = np.pi/6.0

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


def depth_to_pointcloud( depth):
    '''
    Converts depth to pointcloud.
    Arguments:
        * ``depth`` ((W,H) ``float``): Depth data.
    Returns:
        * pc ((3,N) ``float``): Pointcloud.
W        '''
    fy = fx = 0.5 / np.tan(fov * 0.5) # aspectRatio is one.
    

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

def add_noise_to_depth(depth):
    dot_pattern_ = cv2.imread("graspsampler/kinect-pattern_3x3.png", 0)
    # print(dot_pattern_)


    # various variables to handle the noise modelling
    scale_factor  = 200  # converting depth from m to cm 
    focal_length  = 0.5 / np.tan(fov * 0.5)   # focal length of the camera used 
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
    cv2.namedWindow('Adding Kinect Noise', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('Adding Kinect Noise', np.hstack((depth_uint16, noisy_depth)))
    cv2.imshow('Adding Kinect Noise', np.hstack((depth_uint16, noisy_depth)))
    key = cv2.waitKey(1)

    return noisy_depth

def dropout_pointcloud(pointcloud, dropout_ratio=0.875):
    drop_idx = np.where(np.random.random((pointcloud.shape[0]))<=dropout_ratio)[0]
    if len(drop_idx)>0:
        pointcloud[drop_idx,:] = pointcloud[0,:] # set to the first point
    return pointcloud


def depth2noise_pcl(depth,transfered_pose,dropout=0):
    noise_depth = add_noise_to_depth(depth)
    pc =depth_to_pointcloud(noise_depth)

    if dropout > 0:
        pc = dropout_pointcloud(pc,dropout_ratio=dropout)        
    new_pc = trimesh.points.PointCloud(pc)
    new_pc.apply_transform(np.linalg.inv(transfered_pose))
    new_pc = new_pc.vertices.view(np.ndarray)
    
    return new_pc

if __name__ == "__main__":
    import pickle
    import trimesh
    from graspsampler.common import PandaGripper, Scene, Object
    from graspsampler.GraspSampler import PointCloudManager

    pcl_manager = PointCloudManager()

    # with open("depth_maps/ycb/024_bowl.pkl","rb") as handle:
    # with open("depth_maps/ycb/004_sugar_box.pkl","rb") as handle:
    with open("depth_maps/training_data/bottle/bottle000.pkl","rb") as handle:
        data = pickle.load(handle)
        
    with open("/home/user/GRASP/grasp_network/data/pcs/bottle/bottle000.pkl","rb") as handle:
        orig_data = pickle.load(handle)

    print(orig_data.keys())
    exit()

    # print(len(orig_data['pcs']))
    # print(orig_data['camera_poses'][0])


    depth_maps = data["depth_maps"]
    transfered_poses = data["transfered_poses"]
    # print(np.linalg.inv(transfered_poses[0]))
    # exit()

    orig_pc = depth_to_pointcloud(depth_maps[0])
    pc = depth2noise_pcl(depth_maps[0],transfered_poses[0],dropout=0.8)

    count = 0
    for depth_map,transfered_pose in zip(depth_maps,transfered_poses):
        count += 1
        if count > 20:
            break
        test_pc = depth2noise_pcl(depth_map,transfered_pose,dropout=0)
        pc = np.concatenate([pc,test_pc])
    

    pc1 = trimesh.points.PointCloud(vertices=pc)
    pc2 = trimesh.points.PointCloud(vertices=orig_pc)
    transform = np.eye(4)
    transform[0,3] = 0.3
    pc2.apply_transform(transform)
    scene = Scene()
    scene.add_geometry(pc1)
    scene.add_geometry(pc2)

    scene.show()
    