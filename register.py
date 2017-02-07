from camera import Camera_Calibration
from lidar import Lidar_Segment
from sklearn.decomposition import PCA
import numpy as np
import math


class Camera_Lidar_Reg:

    def __init__(self, imgs, lidar_pts):
        # camera corner detection, board recovery and matching
        self.camera = Camera_Calibration(imgs, 0.07699375)
        self.camera.match_cali()
        # lidar points segmentation
        self.lidar = Lidar_Segment(lidar_pts, 40, 0.9)
        self.lidar.segment_data()
        # board centers and normals
        self.camera_board_c = np.zeros((len(self.camera.boards_pts_set), 3))
        self.camera_board_n = np.zeros((len(self.camera.boards_pts_set), 3))
        self.lidar_board_c = np.zeros((len(self.lidar.boards_pts_set), 3))
        self.lidar_board_n = np.zeros((len(self.lidar.boards_pts_set), 3))
        # board probability table for selection
        self.camera_prob_table = []
        self.camera_prob_index = []

    # global registration
    def global_reg(self):
        self.comp_center_normal()
        self.comp_prob()

    # compute normal and center of each planar point cloud
    def comp_center_normal(self):
        pca = PCA(n_components=3)
        for i in xrange(0, len(self.camera.boards_pts_set)):
            self.camera_board_c[i] = np.mean(self.camera.boards_pts_set[i], axis=0)
            pca.fit(self.camera.boards_pts_set[i])
            self.camera_board_n[i] = pca.components_[-1]

        for i in xrange(0, len(self.lidar.boards_pts_set)):
            self.lidar_board_c[i] = np.mean(self.lidar.boards_pts_set[i], axis=0)
            pca.fit(self.lidar.boards_pts_set[i])
            self.lidar_board_n[i] = pca.components_[-1]

    # compute surface triple probability
    def comp_prob(self):
        prob = 0
        for a in xrange(0, len(self.camera.boards_pts_set)):
            for b in xrange(0, len(self.camera.boards_pts_set)):
                if b != a:
                    for c in xrange(0, len(self.camera.boards_pts_set)):
                        if c != a and c != b:
                            self.camera_prob_index.append([a, b, c])
                            self.camera_prob_table.append(prob)
                            prob += math.exp(-self.camera_board_n[a] * self.camera_board_n[b] -
                                             self.camera_board_n[a] * self.camera_board_n[c] -
                                             self.camera_board_n[b] * self.camera_board_n[c])
        self.camera_prob_table = np.array(self.camera_prob_table)
        self.camera_prob_table = self.camera_prob_table / np.sum(self.camera_prob_table) * 1000
