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

with open('camera_points.pickle') as f:
    camera_pts_set = pickle.load(f)
with open('lidar_points.pickle') as f:
    lidar_pts_set = pickle.load(f)
print 'number of camera board', len(camera_pts_set)
print 'number of lidar board', len(lidar_pts_set)


with open('R_set.pickle') as f:
    R = pickle.load(f)
with open('t_set.pickle') as f:
    t = pickle.load(f)

reg = Camera_Lidar_Reg(camera_pts_set, lidar_pts_set, R_set=R, t_set=t)
reg.point_reg_fine_only()
