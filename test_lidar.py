from lidar import Lidar_Segment
import pandas as pd
import numpy as np

lidar_pts = pd.read_csv('./data/lidar.csv', delimiter=' ', header=None).values
Lidar = Lidar_Segment(lidar_pts, 40, 0.9)
Lidar.segment_data()

print len(Lidar.boards_pts_set)
print Lidar.boards_pts_set[0]

pt_output = '/home/bolunzhang/workspace/pcl_cloudViewer/plane1.pcd'
np.savetxt(pt_output, Lidar.boards_pts_set[0])