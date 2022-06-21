# import necessary libs
import pickle
import numpy as np
from dis import dis
from astyx_utils import *
import matplotlib.pyplot as plt
from PIL import Image

# Define constants
DATA_DIR = "./data/"
IMAGE_DIR = DATA_DIR + "image/"
PCD_DIR = DATA_DIR + "pcd/"
BOX2D_DIR = DATA_DIR + "box2d/"
BOX3D_DIR = DATA_DIR + "box3d/"


def main():
    # Getting filename for image, pcd, box2d and box3d
    file_list_image=os.listdir(IMAGE_DIR)
    file_list_no_ex = [x.split('.')[0] for x in file_list_image]
    file_list_pcd = [x+".npy" for x in file_list_no_ex]
    file_list_box2d = [x+"_box2d.pickle" for x in file_list_no_ex]
    file_list_box3d = [x+"_box3d.pickle" for x in file_list_no_ex]

    # Read images from camera
    image_list = []
    for path in file_list_image:
        img = plt.imread(IMAGE_DIR + path)
        image_list.append(img)
    images = np.array(image_list)

    # Read pcd from lidar
    pcd_list = []
    for path in file_list_pcd:
        pcd = np.load(PCD_DIR + path)
        pcd_list.append(pcd)
    pcds = np.array(pcd_list)

    # Read 2d boxes from Detectron
    box2d_list = []
    for path in file_list_box2d:
        with open(BOX2D_DIR + path, 'rb') as handle:
            box2d = pickle.load(handle)
            box2d_list.append(box2d.numpy())
    boxes2d = np.array(box2d_list)
    
    # Read 3d boxes center from OpenPCDet with pointpillar
    box3d_list = []
    for path in file_list_box3d:
        with open(BOX3D_DIR + path, 'rb') as handle:
            box3d = pickle.load(handle)
            box3d_list.append(box3d.numpy()[:,:3])
    boxes3d = np.array(box3d_list)

    boxes3d_to_img = []
    for frame in boxes3d:
        frame_list = lidar2CameraOurs(frame[:, [1, 2, 0]])
        boxes3d_to_img.append(frame_list)


    # know detecting the first frame for number of inliers
    print(f"total number of 3D bounding boxes: {len(boxes3d_to_img[0])}")
    print(f"total number of inliers: {sum(detect_inlier(boxes2d[0], boxes3d_to_img[0]))}")
    print(f"total number of labeled 2D objects: {boxes2d[0].shape[0]}")



def detect_inlier(boxes2d, boxes3d):
    """
    Detect from the 3D boudning boexes if they are inside the 2D bounding boxes
    Args:
    boxes2d : np.array of shape [n, 4]
              boxes2d element [x1, y1, x2, y2]
              x1 -------- y1
               |          |
               |          |
               |          |
              x2 -------- y2
    boxes3d : np.array of shape [m, 2]
              boxes3d element [x, y, z]
    Return:
    list : [boolean] indicating if 3d boxes are inliers
    """
    result = []
    for box3d in boxes3d:
        inlier = False
        for box2d in boxes2d:
            if box3d[0] >= box2d[0] and box3d[0] <= box2d[2] and\
               box3d[1] >= box2d[1] and box3d[1] <= box2d[3]:
               inlier = True
        result.append(inlier)
    return result

if __name__ == "__main__":
    main()