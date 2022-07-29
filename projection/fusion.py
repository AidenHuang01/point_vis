# import necessary libs
import torch
import pickle
import numpy as np
from dis import dis
from astyx_utils import *
import matplotlib.pyplot as plt
from PIL import Image
from projection_utils import *
from tqdm import tqdm

# Define constants
DATA_DIR = "./data/"
IMAGE_DIR = DATA_DIR + "image/"
PCD_DIR = DATA_DIR + "pcd/"
BOX2D_DIR = DATA_DIR + "box2d/"
BOX3D_DIR = DATA_DIR + "box3d/"
PRED_DICT_2D_DIR = DATA_DIR + "pred_dict_2D/"
PRED_DICT_3D_DIR = DATA_DIR + "pred_dict_3D/"

# Getting filename for image, pcd, box2d and box3d
file_list_image=os.listdir(IMAGE_DIR)
file_list_no_ex = [x.split('.')[0] for x in file_list_image]
file_list_pcd = [x+".npy" for x in file_list_no_ex]
file_list_box2d = [x+"_box2d.pickle" for x in file_list_no_ex]
file_list_box3d = [x+"_box3d.pickle" for x in file_list_no_ex]

# Read images from camera
image_list = []
print("reading images")
for path in tqdm(file_list_image):
    img = plt.imread(IMAGE_DIR + path)
    image_list.append(img)
images = np.array(image_list)

# Read pcd from lidar
pcd_list = []
print("reading pcd:")
for path in tqdm(file_list_pcd):
    pcd = np.load(PCD_DIR + path)
    pcd_list.append(pcd)
pcds = np.array(pcd_list)



print("filtering:")
for file in tqdm(file_list_no_ex):
    pred_dict_2D_path = PRED_DICT_2D_DIR + file + "_pred_dict_2D.pickle"
    pred_dict_3D_path = PRED_DICT_3D_DIR + file + "_pred_dict.pickle"
    with open(pred_dict_2D_path, 'rb') as handle:
        pred_dict_2D = pickle.load(handle)
        boxes_2D = pred_dict_2D["boxes"]
        scores_2D = pred_dict_2D["scores"]
        classes_2D = pred_dict_2D["classes"]
    with open(pred_dict_3D_path, 'rb') as handle:
        pred_dict_3D = pickle.load(handle)
        boxes_3D = pred_dict_3D["pred_boxes"]
        scores_3D = pred_dict_3D["pred_scores"]
        labels_3D = pred_dict_3D["pred_labels"]
    boxes_3D_filtered = []
    scores_3D_filtered = []
    labels_3D_filtered = []
    boxes_3D_to_img = lidar2CameraOurs(boxes_3D[:, [1, 2, 0]])
    boxes_3D_to_img = boxes_3D_to_img[:,[1,0]]
    result = detect_inlier(boxes_2D, boxes_3D_to_img)
    pred_dict_3D_filtered = {}
    for i in range(len(result)):
        if result[i]:
            if scores_3D[i] > 0.2:
                boxes_3D_filtered.append(boxes_3D[i])
                scores_3D_filtered.append(scores_3D[i])
                labels_3D_filtered.append(labels_3D[i])
        elif scores_3D[i] > 0.4:
                boxes_3D_filtered.append(boxes_3D[i])
                scores_3D_filtered.append(scores_3D[i])
                labels_3D_filtered.append(labels_3D[i])
    pred_dict_3D_filtered["pred_boxes"] = boxes_3D_filtered
    pred_dict_3D_filtered["pred_scores"] = scores_3D_filtered
    pred_dict_3D_filtered["pred_labels"] = labels_3D_filtered
    output_file_name = "./outputs/" + file + "_pred_dict_filtered.pickle"
    with open(output_file_name, 'wb') as handle:
        pickle.dump(pred_dict_3D_filtered, handle)