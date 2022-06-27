import numpy as np
import torch
from torch import tensor
import open3d
import pickle
import open3d_vis_utils as V
data = np.load('0038_flip_x.npy')
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(x, y, z)
# plt.show()
with open('points.pickle', 'rb') as handle:
    points = pickle.load(handle)
# with open('box.pickle', 'rb') as handle:
#     box = pickle.load(handle)
#     box = box[:3,:]
# with open('labels.pickle', 'rb') as handle:
#     label = pickle.load(handle)
#     label = label[:3]
# with open('scores.pickle', 'rb') as handle:
#     score = pickle.load(handle)
#     score = score[:3]
#     print(label)

with open('0038_pred_dict_filtered.pickle', 'rb') as handle:
    pred_dict = pickle.load(handle)
box = pred_dict['pred_boxes']
score = pred_dict['pred_scores']
label = pred_dict['pred_labels']
box = torch.stack(box)
score = torch.stack(score)
label = torch.stack(label)


V.draw_scenes(points, ref_boxes=box, ref_scores=score, ref_labels=label)