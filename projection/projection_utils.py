import numpy as np


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
            if box3d[0] >= box2d[0] * 0.8 and box3d[0] <= box2d[2] * 1.2 and\
               box3d[1] >= box2d[1] * 0.8 and box3d[1] <= box2d[3] * 1.2:
               inlier = True
        result.append(inlier)
    return result