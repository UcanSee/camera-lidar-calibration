from camera import Camera_Calibration
from lidar import Lidar_Segment
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import numpy as np
import math
import cv2

class Camera_Lidar_Reg:

    def __init__(self, camera_pts_set, lidar_pts_set, tau_score=1.5, tau_comb=25, R_set=[], t_set=[]):
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
        self.camera_prob_dict = {}
        self.lidar_prob_index = []
        self.lidar_prob_dict = {}
        # parameters for global registration
        self.tau_score = tau_score
        self.tau_comb = tau_comb
        # global registration result
        self.R_set = R_set
        self.t_set = t_set
        self.score_set = []

    # global registration
    def point_reg(self):
        # compute surface normals and surface centroids
        self.comp_center_normal()
        # compute probability table for surface triple selection
        self.comp_prob()
        # set up nearest neighbor map for score function
        self.set_lidar_nn()
        # global registration
        self.global_reg()
        # fine registration by ICP
        self.fine_reg()

    # global registration
    def point_reg_fine_only(self):
        # set up nearest neighbor map for score function
        self.set_lidar_nn()
        # fine registration by ICP
        self.fine_reg()

    # compute normal and center of each planar point cloud
    def comp_center_normal(self):
        pca = PCA(n_components=3)
        for i in xrange(0, len(self.camera_pts_set)):
            self.camera_board_c[i] = np.mean(self.camera_pts_set[i], axis=0)
            pca.fit(self.camera_pts_set[i])
            normal_vec = pca.components_[-1]
            self.camera_board_n[i] = -1 * np.sign(normal_vec[2]) * normal_vec

        for i in xrange(0, len(self.lidar_pts_set)):
            self.lidar_board_c[i] = np.mean(self.lidar_pts_set[i], axis=0)
            pca.fit(self.lidar_pts_set[i])
            normal_vec = pca.components_[-1]
            self.lidar_board_n[i] = -1 * np.sign(normal_vec[0]) * normal_vec

    # compute camera checkerboard surface triple probability
    def comp_prob(self):
        # camera checkerboard probability table and index
        for a in xrange(0, len(self.camera_pts_set)):
            for b in xrange(0, len(self.camera_pts_set)):
                if b != a:
                    for c in xrange(0, len(self.camera_pts_set)):
                        if c != a and c != b:
                            self.camera_prob_dict[(a, b, c)] = len(self.camera_prob_index)
                            self.camera_prob_index.append([a, b, c])
                            prob = math.exp(- abs(self.camera_board_n[a].dot(self.camera_board_n[b]))
                                            - abs(self.camera_board_n[a].dot(self.camera_board_n[c]))
                                            - abs(self.camera_board_n[b].dot(self.camera_board_n[c])))
                            self.camera_prob_table.append(prob)
        self.camera_prob_table = np.array(self.camera_prob_table)
        self.camera_prob_table /= np.sum(self.camera_prob_table)

        # lidar checkerboard probability table and index
        for a in xrange(0, len(self.lidar_pts_set)):
            for b in xrange(0, len(self.lidar_pts_set)):
                if b != a:
                    for c in xrange(0, len(self.lidar_pts_set)):
                        if c != a and c != b:
                            self.lidar_prob_dict[(a, b, c)] = len(self.lidar_prob_index)
                            self.lidar_prob_index.append([a, b, c])

        # traversal matrix
        self.global_search_mat = np.zeros((len(self.camera_prob_index), len(self.lidar_prob_index)), dtype=bool)

    # set up nearest neighbor map for score function
    def set_lidar_nn(self):
        temp_lidar_pts = []
        for lidar_set in self.lidar_pts_set:
            for i in xrange(len(lidar_set)):
                temp_lidar_pts.append(lidar_set[i])
        temp_lidar_pts = np.array(temp_lidar_pts)
        self.lidar_nn = KDTree(temp_lidar_pts, leaf_size=30, metric='euclidean')

    # global registration from camera to range manual assigned a combination
    def global_reg_test(self):
        while True:
            # random sample surface triples
            s_c = [2, 9, 3]
            s_r = [11, 1, 14]

            # find optimal rotation from camera surface to range surface
            cov_mat = np.zeros((3, 3))
            print('normal 1', self.camera_board_n[s_c[0]:s_c[0] + 1, :].dot(self.lidar_board_n[s_r[0]:s_r[0] + 1, :].T))
            print('normal 2', self.camera_board_n[s_c[1]:s_c[1] + 1, :].dot(self.lidar_board_n[s_r[1]:s_r[1] + 1, :].T))
            print('normal 3', self.camera_board_n[s_c[2]:s_c[2] + 1, :].dot(self.lidar_board_n[s_r[2]:s_r[2] + 1, :].T))
            cov_mat += self.camera_board_n[s_c[0]:s_c[0] + 1, :].T.dot(self.lidar_board_n[s_r[0]:s_r[0] + 1, :])
            cov_mat += self.camera_board_n[s_c[1]:s_c[1] + 1, :].T.dot(self.lidar_board_n[s_r[1]:s_r[1] + 1, :])
            cov_mat += self.camera_board_n[s_c[2]:s_c[2] + 1, :].T.dot(self.lidar_board_n[s_r[2]:s_r[2] + 1, :])
            U, S, V = np.linalg.svd(cov_mat)
            R = V.T.dot(U.T)

            # find optimal translation by minimizing point-to-plane distance A*t=B
            A = np.zeros((3, 3))
            B = np.zeros((3, 1))
            for i in xrange(len(s_r)):
                n_mat = self.lidar_board_n[s_r[i]:s_r[i] + 1, :].T.dot(self.lidar_board_n[s_r[i]:s_r[i] + 1, :])
                A += n_mat
                B += n_mat.dot(self.lidar_board_c[s_r[i]:s_r[i] + 1, :].T -
                               R.dot(self.camera_board_c[s_c[i]:s_c[i] + 1, :].T))
            t = np.linalg.inv(A).dot(B)

            # compute RT score
            score_temp = self.global_reg_score(s_c, R, t)

            print(R)
            print(t)
            print(score_temp)
            break

    # global registration from camera to range
    def global_reg(self):
        high_score = -float('Inf')
        count = 0
        camera_num = len(self.camera_prob_index)
        lidar_num = len(self.lidar_prob_index)
        while count < 50000:
            break_flag = False
            # random sample surface triples
            s_c_idx = np.random.choice(camera_num, 1, p=self.camera_prob_table)[0]
            s_r_idx = np.random.choice(lidar_num, 1)[0]
            if self.global_search_mat[s_c_idx, s_r_idx]:
                continue
            else:
                s_c = self.camera_prob_index[s_c_idx]
                s_r = self.lidar_prob_index[s_r_idx]
                for i in xrange(3):
                    for j in xrange(3):
                        if j != i:
                            for k in xrange(3):
                                if k != j and k != i:
                                    cam_idx = self.camera_prob_dict[(s_c[i], s_c[j], s_c[k])]
                                    lid_idx = self.lidar_prob_dict[(s_r[i], s_r[j], s_r[k])]
                                    self.global_search_mat[cam_idx, lid_idx] = True

            # find optimal rotation from camera surface to range surface
            cov_mat = np.zeros((3, 3))
            for i in xrange(len(s_c)):
                n_c = self.camera_board_n[s_c[i]:s_c[i]+1, :]
                n_r = self.lidar_board_n[s_r[i]:s_r[i]+1, :]
                # cov_mat += -1 * np.sign(n_c.dot(n_r.T)) * n_c.T.dot(n_r)
                cov_mat += n_c.T.dot(n_r)
            U, S, V = np.linalg.svd(cov_mat)
            R = V.T.dot(U.T)

            # find optimal translation by minimizing point-to-plane distance A*t=B
            A = np.zeros((3, 3))
            B = np.zeros((3, 1))
            for i in xrange(len(s_r)):
                n_mat = self.lidar_board_n[s_r[i]:s_r[i]+1, :].T.dot(self.lidar_board_n[s_r[i]:s_r[i]+1, :])
                A += n_mat
                B += n_mat.dot(self.lidar_board_c[s_r[i]:s_r[i]+1, :].T -
                               R.dot(self.camera_board_c[s_c[i]:s_c[i]+1, :].T))
            t = np.linalg.inv(A).dot(B)

            # compute RT score
            score_temp = self.global_reg_score(s_c, R, t)

            if score_temp > self.tau_score * high_score:
                self.R_set.append(R)
                self.t_set.append(t)
                self.score_set.append(score_temp)

            if score_temp > high_score:
                high_score = score_temp
                temp_score_set =[]
                temp_R_set = []
                temp_t_set = []
                for i in xrange(len(self.score_set)):
                    if self.score_set[i] > self.tau_score * high_score:
                        temp_score_set.append(self.score_set[i])
                        temp_R_set.append(self.R_set[i])
                        temp_t_set.append(self.t_set[i])
                self.score_set = temp_score_set
                self.R_set = temp_R_set
                self.t_set = temp_t_set

            # check termination criteria
            # a. enough good transformation
            print len(self.score_set)
            if len(self.score_set) >= self.tau_comb:
                break_flag = True

            # b. processed all possible combination
            #print camera_num * lidar_num - self.global_search_mat.sum()
            if self.global_search_mat.sum() == camera_num * lidar_num:
                break_flag = True

            # prevent premature break before finding a usable R and t
            if break_flag and count > 10000:
                break
            count += 1

        # debug
        for i in xrange(len(self.R_set)):
            print "optimal cost", 1.0 * self.score_set[i]
            print "optimal R: ", self.R_set[i]
            print "optimal t: ", self.t_set[i]
            print "optimal actual cost: ", 1.0 * self.score_set[i] + 0.7 * np.linalg.norm(self.t_set[i])

    # cost function for global registration
    def global_reg_score(self, s_c, R, t):
        score = 0
        for i in xrange(len(s_c)):
            pt_cam_tilda = (R.dot(self.camera_board_c[s_c[i]:s_c[i]+1, :].T) + t).T
            (score_temp, idx) = self.lidar_nn.query(pt_cam_tilda, k=1, return_distance=True)
            score += score_temp[0][0]
        return -1.0 * score - 0.7 * np.linalg.norm(t)

    # icp
    def fine_reg(self, num_iter=1):
        R_set_refine = []
        t_set_refine = []
        cam_pts = []
        for cam_set in self.camera_pts_set:
            for i in xrange(len(cam_set)):
                cam_pts.append(cam_set[i])
        cam_pts = np.array(cam_pts)
        lidar_pts = []
        for lidar_set in self.lidar_pts_set:
            for i in xrange(len(lidar_set)):
                lidar_pts.append(lidar_set[i])
        lidar_pts = np.array(lidar_pts)

        for i in xrange(len(self.R_set)):
            # initial pose
            R = self.R_set[i]
            t = self.t_set[i]
            print 'initial R: ', R
            print 'initial t: ', t
            cam_pts_proj = (R.dot(cam_pts.T) + t).T
            for j in xrange(num_iter):
                (distances, idx) = self.lidar_nn.query(cam_pts_proj, k=1, return_distance=True)
                if j == 0:
                    print "inital squared error %f" % np.square(distances).sum()
