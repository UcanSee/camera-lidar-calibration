from corner import Corner_Detection
from checkerboard import Board_Recovery
import cv2
import math

img = []
img.append(cv2.imread('/home/bolunzhang/Downloads/gray_calibration/calibration1/frame0004.jpg'))
img.append(cv2.imread('./data/right.jpg'))
corner = Corner_Detection(img[0], 0.02)
corner.find_corners()
board = Board_Recovery(corner.corners_pts, corner.corners_v1, corner.corners_v2)
board.find_boards()

I_1 = img[0]
for i in xrange(0, len(board.chessboards)):
    b_t = board.chessboards[i]
    for h in xrange(0, len(b_t)):
        for w in xrange(0, len(b_t[0])):
            x = int(math.ceil(board.c1[b_t[h, w]][0]))
            y = int(math.ceil(board.c1[b_t[h, w]][1]))
            I_1[y-2:y+2, x-2:x+2] = [0, 0, 255]
cv2.imshow('left', I_1)
cv2.waitKey(0)
