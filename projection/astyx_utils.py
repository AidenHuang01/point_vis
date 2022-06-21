import sys
sys.path.append("../")
sys.path.append("../utils/")
sys.path.append("./utils/")
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from pyquaternion import Quaternion
import numpy as np
import matplotlib.pyplot as plt
import json
import copy

class Astyx_Data:
    def __init__(self, astyx_folder = "./astyx/"):
        
        self.astyx_camera_folder = astyx_folder + "camera_front/"
        self.astyx_radar_folder = astyx_folder + "radar_6455/"
        self.astyx_calib_folder = astyx_folder + "calibration/"
        self.astyx_label_folder = astyx_folder + "groundtruth_obj3d/"
        self.astyx_lidar_folder = astyx_folder + "lidar_vlp16/"

        self.radar = os.listdir(self.astyx_radar_folder)
        self.camera = os.listdir(self.astyx_camera_folder)
        self.calib = os.listdir(self.astyx_calib_folder)
        self.label = os.listdir(self.astyx_label_folder)
        self.lidar = os.listdir(self.astyx_lidar_folder)
    
    def get_folders(self):
        return self.radar, self.camera, self.calib, self.label
    
    def get_file_name(self, idx):
        return str(format(idx, '06d'))

    def get_camera(self, file_name):
        return plt.imread(self.astyx_camera_folder+file_name+".jpg")
    
    def get_radar(self, file_name):
        return np.loadtxt(self.astyx_radar_folder+file_name+".txt", skiprows = 2)
    
    def get_lidar(self, file_name):
        return np.loadtxt(self.astyx_lidar_folder+file_name+".txt", skiprows = 1)
    
    def get_calib_data(self):
        file_read = open(self.astyx_calib_folder+str(format(0, '06d'))+".json")
        calib_data = json.load(file_read)
        file_read.close()
        return calib_data

    def read_data(self, idx):

        file_name = self.get_file_name(idx)
        
        im = self.get_camera(file_name)
        radardata = self.get_radar(file_name)
        lidardata = self.get_lidar(file_name)
        calibdata = self.get_calib_data(file_name)
        label_box = self.get_astyx_labels(file_name)

        return im, radardata, lidardata, calibdata, label_box


    def get_astyx_labels(self, file_name):
        '''
        Return the list of labels stored in w,h,l,x,y,z,theta format
        '''

        with open(self.astyx_label_folder+file_name+".json") as f:
            labels = json.load(f)

        label_box = []    
        for item in labels['objects']:    
            quat = Quaternion(item['orientation_quat']) 
            theta = quat.angle*(1 if item['orientation_quat'][-1]>0 else -1)
            center = item['center3d']
            dimension = item['dimension3d']
            # if(np.linalg.norm(center)<100):
            label_box.append(np.hstack(([dimension[1],dimension[2],\
                dimension[0]],center,theta)))  # w,h,l,x,y,z,theta  
        label_box = np.array(label_box)
        return label_box

    def filter_astyx_radar(self, radardata):
        radardata = radardata[(radardata[:,1]>-50)*(radardata[:,1]<50)]
        radardata = radardata[(radardata[:,0]>0)]#*(radardata[:,0]<100)]
        radardata = radardata[(radardata[:,2]>-1)*(radardata[:,2]<3)]
        return radardata
    
    def get_calibration_details(self):
        # Camera parameters
        calib_data = self.get_calib_data()
        camera_intrinsic = np.array(calib_data['sensors'][2]['calib_data']['K'])
        camera_extrinsic = np.array(calib_data['sensors'][2]['calib_data']\
            ['T_to_ref_COS'])
        
        # Radar parameters
        radar_extrinsic = np.array(calib_data['sensors'][0]['calib_data']\
            ['T_to_ref_COS'])
        lidar_extrinsic = np.array(calib_data['sensors'][1]['calib_data']\
            ['T_to_ref_COS'])
        
        params = {}
        params['camera_intrinsics'] = camera_intrinsic
        params['camera_extrinsics'] = camera_extrinsic
        params['radar_extrinsics'] = radar_extrinsic
        params['lidar_extrinsics'] = lidar_extrinsic

        return params


class astyx_projection():
    
    def __init__(self, params):
        
        self.camera_intrinsics = params['camera_intrinsics']
        self.camera_extrinsics = params['camera_extrinsics']
        self.radar_extrinsics = params['radar_extrinsics']
        self.lidar_extrinsics = params['lidar_extrinsics']
        
#     def pc_filter(self, pc, [xmin, xmax, ymin, ymax, zmin, zmax]):
#         pc = pc[(pc[:,1]>ymin)*(pc[:,1]<ymax)]
#         pc = pc[(pc[:,0]>xmin)*(pc[:,0]<xmax)]
#         pc = pc[(pc[:,2]>zmin)*(pc[:,2]<zmax)]
        
        
    def radar2CameraAstyx(self, radar_pc):
        ones = np.ones((radar_pc.shape[0],1))
        radar_pc_pad = np.hstack((radar_pc,ones))
        
        radar2world = np.matmul(self.radar_extrinsics,radar_pc_pad.T)

        world2camera = np.matmul(np.linalg.inv(self.camera_extrinsics),radar2world)

        camera_intrinsic_padded = np.hstack((self.camera_intrinsics,np.zeros((3,1))))
        camera2image = np.matmul(camera_intrinsic_padded,world2camera)

        image_coords = np.vstack((camera2image[0,:]/camera2image[2,:],camera2image[1,:]/camera2image[2,:])).T
        
        return image_coords

    def lidar2CameraAstyx(self, lidar_pc):
        ones = np.ones((lidar_pc.shape[0],1))
        lidar_pc_pad = np.hstack((lidar_pc,ones))
        
        lidar2world = np.matmul(self.lidar_extrinsics,lidar_pc_pad.T)

        world2camera = np.matmul(np.linalg.inv(self.camera_extrinsics),lidar2world)

        camera_intrinsic_padded = np.hstack((self.camera_intrinsics,np.zeros((3,1))))
        camera2image = np.matmul(camera_intrinsic_padded,world2camera)

        image_coords = np.vstack((camera2image[0,:]/camera2image[2,:],camera2image[1,:]/camera2image[2,:])).T
        
        return image_coords

    def get_ray_angle(self, image_point):
        if image_point.ndim == 1:
            image_point = np.expand_dims(image_point,0)
        
        n_points = image_point.shape[0]
        image_point = np.hstack((image_point,np.ones((n_points,1))))
        image_test_coords = image_point.T
        
        camera_intrinsic_inv = np.vstack((np.linalg.inv(self.camera_intrinsics),np.zeros((1,3))))
        
        image2camera = np.matmul(camera_intrinsic_inv,image_test_coords)
        camera2world = np.matmul(np.linalg.inv(self.camera_extrinsics[:3,:3]).T,image2camera[:3,:])
        
        camera2world = camera2world.T

        return np.expand_dims(np.arctan(camera2world[:,1]/camera2world[:,0]),1)

def lidar2CameraOurs(radar_pc):
    ''' 
    This is the projection code for our dataset to project pointcloud onto the camera plane for the mask based clustering
    input: [Hor, height, depth]
    '''
    image_coords = np.zeros((radar_pc.shape[0],2))
    for pidx, points in enumerate(radar_pc):        
        point = points[:3]
        x = point[0] / point[2]
        y = point[1] / point[2]
        coeffs = [0,0,0,0,0]
        fx = 1383.08288574219#595.037*(1920/620)
        fy = 1381.68029785156#595.037*(1080/480)
        # ppx = 318.33
        # ppy = 237.47
        ppx = 945.295715332031
        ppy = 530.814331054688
        r2  = x*x + y*y
        f = 1 + coeffs[0]*r2 + coeffs[1]*r2*r2 + coeffs[4]*r2*r2*r2
        x *= f
        y *= f
        dx = x + 2*coeffs[2]*x*y + coeffs[3]*(r2 + 2*x*x)
        dy = y + 2*coeffs[3]*x*y + coeffs[2]*(r2 + 2*y*y)
        x = dx
        y = dy
        pixel = [x * fx + ppx,ppy - y * fy]
        image_coords[pidx] = pixel
    return image_coords