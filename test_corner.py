from corner import Corner_Detection
from checkerboard import Board_Recovery
from camera import Camera_Calibration
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = []
img.append(cv2.imread('./data/cam1.jpg'))
img.append(cv2.imread('./data/cam2.jpg'))
camera = Camera_Calibration(img, 0.08)  # 0.08m distance from one corner to next corner
camera.match_cali()
print camera.intrinsic
print camera.rotation
print camera.translation

