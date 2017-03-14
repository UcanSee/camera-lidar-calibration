from camera import Camera_Calibration
from lidar import Lidar_Segment
from register import Camera_Lidar_Reg
import cv2
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

'''
# obtain camera checkerboard 3d points
imgs = []
imgs.append(cv2.imread('./data/left.jpg'))
imgs.append(cv2.imread('./data/right.jpg'))
# camera corner detection, board recovery and matching
camera = Camera_Calibration(imgs, 0.08)
camera.match_cali()
print "number of camera boards %d \n" % (len(camera.boards_pts_set))
print camera.intrinsic
print camera.rotation
print camera.translation
with open('camera_points.pickle', 'w') as f:
    pickle.dump(camera.boards_pts_set, f)
'''
'''
# obtain lidar checkboard 3d points
lidar_pts = pd.read_csv('./data/lidar.csv', delimiter=' ', header=None).values
# lidar points segmentation
lidar = Lidar_Segment(lidar_pts, 50, 0.9)
lidar.segment_data()
print "number of lidar boards %d \n" % (len(lidar.boards_pts_set))
with open('lidar_points.pickle', 'w') as f:
    pickle.dump(lidar.boards_pts_set, f)
'''
with open('camera_points.pickle') as f:
    camera_pts_set = pickle.load(f)
with open('lidar_points.pickle') as f:
    lidar_pts_set = pickle.load(f)

# visualize 3d points
print "number of camera boards %d \n" % (len(camera_pts_set))
print "number of lidar boards %d \n" % (len(lidar_pts_set))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in xrange(0, len(camera_pts_set)):
    ax.scatter(camera_pts_set[i][:, 0], camera_pts_set[i][:, 1], camera_pts_set[i][:, 2])
    ax.text(np.mean(camera_pts_set[i][:, 0]), np.mean(camera_pts_set[i][:, 1]), np.mean(camera_pts_set[i][:, 2]),
            str(i), color=[1.0, 0.0, 0.0])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
colors = cm.rainbow(np.linspace(0, 1, len(lidar_pts_set)))
for i in xrange(0, len(lidar_pts_set)):
    if len(lidar_pts_set[i]) < 400:
        ax2.scatter(lidar_pts_set[i][:, 0], lidar_pts_set[i][:, 1], lidar_pts_set[i][:, 2])
        ax2.text(np.mean(lidar_pts_set[i][:, 0]), np.mean(lidar_pts_set[i][:, 1]), np.mean(lidar_pts_set[i][:, 2]),
            str(i), color=[1.0, 0.0, 0.0])
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
plt.show()

reg = Camera_Lidar_Reg(camera_pts_set, lidar_pts_set)
reg.point_reg()
