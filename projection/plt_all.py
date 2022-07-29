import pickle
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from projection_utils import *
pred_dict_2D_path = '/home/yucheng/storage/data/0720/test_gilman_04/2D_boxes/0022.pickle'
box_3d_dir_path = '/home/yucheng/storage/data/0720/test_gilman_04/3D_boxes/0022_pred_dict.pickle'
img_path = '/home/yucheng/storage/data/0720/test_gilman_04/img_detection/0022.png'


with open(box_3d_dir_path, 'rb') as handle:
    pred_dict_3D = pickle.load(handle)
    boxes_3D = pred_dict_3D["pred_boxes"]
    scores_3D = pred_dict_3D["pred_scores"]
    labels_3D = pred_dict_3D["pred_labels"]

with open(pred_dict_2D_path, 'rb') as handle:
    pred_dict_2D = pickle.load(handle)
    boxes_2D = pred_dict_2D["boxes"]
    scores_2D = pred_dict_2D["scores"]
    classes_2D = pred_dict_2D["classes"]

boxes_3D_to_img = lidar2CameraOurs(boxes_3D[:, [1, 2, 0]])[:,[1,0]]

img = Image.open(img_path).resize((1920, 1080))
img_matrix = np.asanyarray(img)
plt.imshow(img_matrix)

print(boxes_3D_to_img[2:3, :])

res = detect_inlier(boxes_2D, boxes_3D_to_img[2:3, :])

print(boxes_2D.astype(int))

print(res)

for i in range(boxes_3D_to_img.shape[0]):
    x = boxes_3D_to_img[i][1]
    y = boxes_3D_to_img[i][0]
    if x >= 0 and x <= 1080 and y >= 0 and y <= 1920:
        # plt.text(x, y, i, c='r')
        plt.scatter(x, y, 50, c="r", marker="+") # plot markers

plt.scatter(506, 502, 50, c="g", marker="+")
plt.scatter(820, 649, 50, c="g", marker="+")
plt.savefig("save.png")