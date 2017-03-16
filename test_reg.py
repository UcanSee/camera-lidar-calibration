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

# obtain camera checkerboard 3d points
imgs = []
imgs.append(cv2.imread('./data/left_0303_n.jpg'))
imgs.append(cv2.imread('./data/right_0303_n.jpg'))
'''
# camera corner detection, board recovery and matching
camera = Camera_Calibration(imgs, 0.0774)
camera.match_cali()
print "number of camera boards %d \n" % (len(camera.boards_pts_set))
print camera.intrinsic
print camera.rotation
print camera.translation
print camera.distortion
with open('camera_points.pickle', 'w') as f:
    pickle.dump(camera.boards_pts_set, f)
'''
# obtain lidar checkboard 3d points
lidar_pts = pd.read_csv('./data/lidar_0303_n.csv', delimiter=' ', header=None).values

# lidar points segmentation
lidar = Lidar_Segment(lidar_pts, 40, 0.9, 400)
lidar.segment_data()
print "number of lidar boards %d \n" % (len(lidar.boards_pts_set))
with open('lidar_points.pickle', 'w') as f:
    pickle.dump(lidar.boards_pts_set, f)

with open('camera_points.pickle') as f:
    camera_pts_set = pickle.load(f)
with open('lidar_points.pickle') as f:
    lidar_pts_set = pickle.load(f)

print 'number of camera board', len(camera_pts_set)
print 'number of lidar board', len(lidar_pts_set)

# visualize 3d points
print "number of camera boards %d \n" % (len(camera_pts_set))
print "number of lidar boards %d \n" % (len(lidar_pts_set))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in xrange(0, len(camera_pts_set)):
    ax.scatter(camera_pts_set[i][:, 0], camera_pts_set[i][:, 1], camera_pts_set[i][:, 2])
    ax.text(np.mean(camera_pts_set[i][:, 0]), np.mean(camera_pts_set[i][:, 1]), np.mean(camera_pts_set[i][:, 2]),
            str(i), color=[1.0, 0.0, 0.0], fontsize=15)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


fig = plt.figure()
ax2 = fig.add_subplot(111,  projection='3d')
colors = cm.rainbow(np.linspace(0, 1, len(lidar_pts_set)))
#l = [2, 3, 8, 14, 16, 18, 21]
l = []
for i in xrange(0, len(lidar_pts_set)):
    if 400 < len(lidar_pts_set[i]) < 1000 and i not in l:
        print len(lidar_pts_set[i])
        ax2.scatter(lidar_pts_set[i][:, 0], lidar_pts_set[i][:, 1], lidar_pts_set[i][:, 2])
        ax2.text(np.mean(lidar_pts_set[i][:, 0]), np.mean(lidar_pts_set[i][:, 1]), np.mean(lidar_pts_set[i][:, 2]),
            str(i), color=[1.0, 0.0, 0.0], fontsize=20)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
plt.show()

reg = Camera_Lidar_Reg(camera_pts_set, lidar_pts_set, lidar_pts)
reg.point_reg()
with open('R_set.pickle', 'w') as f:
    pickle.dump(reg.R_set, f)
with open('t_set.pickle', 'w') as f:
    pickle.dump(reg.t_set, f)
trans_mat_set = reg.trans_mat_set
'''
with open('R_set.pickle') as f:
    R = pickle.load(f)
with open('t_set.pickle') as f:
    t = pickle.load(f)

reg = Camera_Lidar_Reg(camera_pts_set, lidar_pts_set, lidar_pts, R_set=R, t_set=t)
reg.point_reg_fine_only()
trans_mat_set = reg.trans_mat_set
'''
K = np.array([[6.47816e2, 0.0, 7.71052e2], [0.0, 6.45650e2, 4.33588e2], [0.0, 0.0, 1.0]])
# visualize the corners found on image
l = []
for set_no in xrange(len(trans_mat_set)):
    I_1 = np.array(imgs[0], copy=True)
    for no_board in xrange(0, len(lidar_pts_set)):
        if no_board not in l:
            for i in xrange(0, len(lidar_pts_set[no_board])):
                x = lidar_pts_set[no_board][i, 0]
                y = lidar_pts_set[no_board][i, 1]
                z = lidar_pts_set[no_board][i, 2]
                lidar_pt = np.array([[x], [y], [z], [1]])

                proj_lidar_pt = np.linalg.inv(trans_mat_set[set_no]).dot(lidar_pt)
                proj_lidar_pt = proj_lidar_pt[0: 3, :]
                tmp = K.dot(proj_lidar_pt)
                x = int(tmp[0] / tmp[2])
                y = int(tmp[1] / tmp[2])
                I_1[y-1:y+1, x-1:x+1] = [0, 0, 255]
    cv2.imshow(str(reg.error_set[set_no]), I_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
