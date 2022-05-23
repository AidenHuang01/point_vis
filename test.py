import numpy as np
from torch import tensor
import open3d
import pickle
import open3d_vis_utils as V
data = np.load('0038.npy')
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
    points = pickle.load(handle).cpu()
box = tensor([[19.8113, -2.1688, -0.7888,  3.6943,  1.6010,  1.5460,  3.3396],[41.7992, -2.3666, -0.0635,  3.8994,  1.6166,  1.5087,  3.0896]]).cpu()
print(points)
V.draw_scenes(points, ref_boxes=box)