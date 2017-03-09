from camera import Camera_Calibration
from lidar import Lidar_Segment
from sklearn.decomposition import PCA
import numpy as np
import math


class Camera_Lidar_Reg:

    def __init__(self, camera_pts_set, lidar_pts_set):
        # camera corner detection, board recovery and matching
        self.camera_pts_set = camera_pts_set
        self.lidar_pts_set = lidar_pts_set
        # board centers and normals
        self.camera_board_c = np.zeros((len(self.camera_pts_set), 3))
        self.camera_board_n = np.zeros((len(self.camera_pts_set), 3))
        self.lidar_board_c = np.zeros((len(self.lidar_pts_set), 3))
        self.lidar_board_n = np.zeros((len(self.lidar_pts_set), 3))
        # board probability table for selection
        self.camera_prob_table = []
        self.camera_prob_index = []

    # global registration
    def point_reg(self):
        # compute surface normals and surface centroids
        self.comp_center_normal()
        # compute probability table for surface triple selection
        self.comp_prob()
        # global registration
        self.global_reg()

    # compute normal and center of each planar point cloud
    def comp_center_normal(self):
        pca = PCA(n_components=3)
        for i in xrange(0, len(self.camera_pts_set)):
            self.camera_board_c[i] = np.mean(self.camera_pts_set[i], axis=0)
            pca.fit(self.camera_pts_set[i])
            self.camera_board_n[i] = pca.components_[-1]

        for i in xrange(0, len(self.lidar_pts_set)):
            self.lidar_board_c[i] = np.mean(self.lidar_pts_set[i], axis=0)
            pca.fit(self.lidar_pts_set[i])
            self.lidar_board_n[i] = pca.components_[-1]

    # compute camera checkerboard surface triple probability
    def comp_prob(self):
        for a in xrange(0, len(self.camera_pts_set)):
            for b in xrange(0, len(self.camera_pts_set)):
                if b != a:
                    for c in xrange(0, len(self.camera_pts_set)):
                        if c != a and c != b:
                            self.camera_prob_index.append([a, b, c])
                            prob = math.exp(- abs(self.camera_board_n[a].dot(self.camera_board_n[b]))
                                            - abs(self.camera_board_n[a].dot(self.camera_board_n[c]))
                                            - abs(self.camera_board_n[b].dot(self.camera_board_n[c])))
                            self.camera_prob_table.append(prob)
        self.camera_prob_table = np.array(self.camera_prob_table)
        self.camera_prob_table /= np.sum(self.camera_prob_table)

    # global registration from camera to range
    def global_reg(self):
        while True:
            # random select surface triples
            s_c = self.camera_prob_index[np.random.choice(len(self.camera_prob_table), 1,
                                                          p=self.camera_prob_table)[0]]
            s_r = np.random.choice(len(self.lidar_board_c), 3, replace=False).tolist()
            # find optimal rotation from camera surface to range surface
            cov_mat = np.zeros((3, 3))
            for i in xrange(len(s_c)):
                cov_mat += self.camera_board_n[s_c[i]:s_c[i]+1, :].T.dot(self.lidar_board_n[s_r[i]:s_r[i]+1, :])
            U, S, V = np.linalg.svd(cov_mat)
            R = V.dot(U.T)
            # find optimal translation by minimizing point-to-plane distance A*t=B
            A = np.zeros((3, 3))
            B = np.zeros((3, 1))
            for i in xrange(len(s_r)):
                n_mat = self.lidar_board_n[s_r[i]:s_r[i]+1, :].T.dot(self.lidar_board_n[s_r[i]:s_r[i]+1, :])
                A += n_mat
                B += n_mat.dot(self.lidar_board_c[s_r[i]:s_r[i]+1, :].T -
                               R.dot(self.camera_board_c[s_c[i]:s_c[i]+1, :].T))
            t = np.linalg.inv(A).dot(B)
            print(R)
            print(t)
            break
