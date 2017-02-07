from register import Camera_Lidar_Reg
import cv2
import pandas as pd

imgs = []
imgs.append(cv2.imread('./data/left.jpg'))
imgs.append(cv2.imread('./data/right.jpg'))
lidar_pts = pd.read_csv('./data/lidar.csv', delimiter=' ', header=None).values
reg = Camera_Lidar_Reg(imgs, lidar_pts)