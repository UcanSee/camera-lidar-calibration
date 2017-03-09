from corner import Corner_Detection
from checkerboard import Board_Recovery
from camera import Camera_Calibration
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = []
img.append(cv2.imread('./data/left_0303_n.jpg'))
img.append(cv2.imread('./data/right_0303_n.jpg'))
camera = Camera_Calibration(img, 0.08)  # 0.08m distance from one corner to next corner
# camera = Camera_Calibration(img, 0.08)  # 0.08m distance from one corner to next corner
camera.match_cali()
print camera.intrinsic
print camera.translation

'''
n_pts = 0
lidar_file = open('/home/bolunzhang/workspace/pcl_cloudViewer/camera_plane.pcd', 'w')
for i in xrange(len(camera.boards_pts_set)):
    b_s = camera.boards_pts_set[i]
    for j in xrange(len(b_s)):
        n_pts += 1
        lidar_file.write(str(b_s[j, 0]) + " " + str(b_s[j, 1]) + " " + str(b_s[j, 2]) + "\n")
print n_pts
'''