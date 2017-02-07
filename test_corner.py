from corner import Corner_Detection
from checkerboard import Board_Recovery
from camera import Camera_Calibration
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = []
img.append(cv2.imread('./data/left.jpg'))
img.append(cv2.imread('./data/right.jpg'))
camera = Camera_Calibration(img, 0.07699375)  # 0.08m distance from one corner to next corner
# camera = Camera_Calibration(img, 0.08)  # 0.08m distance from one corner to next corner
camera.match_cali()
print camera.boards_pts_set[0]
