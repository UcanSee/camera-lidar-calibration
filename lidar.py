import numpy as np
import scipy.spatial as spatial
import math
import cv2
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
'''
Implement range data segmentation into planes potentially corresponding to checkerboards
'''


class Lidar_Segment:

    def __init__(self, lidar_pts, n_nbrs, tau_seg, min_size):
        # range data [[x, y, z]]
        self.lidar_pts = lidar_pts
        # number of neighbors used in finding normals and growing boards
        self.n_nbrs = n_nbrs
        # neighbors of each point
        self.nbrs_indices = []
        # normal vector of each range data
        self.lidar_normal = []
        # board set Pj_r, each set is a point index list
        self.boards_set = []
        # board set Pj_r, each set is a 3d point list
        self.boards_pts_set = []
        # threshold for segmentation check if two points are planar
        self.tau_seg = tau_seg
        # minimize size threshold for a board
        self.min_size = min_size

    # main function of range data segmentation
    def segment_data(self):
        self.comp_normal()
        self.find_board()
        self.filter_board(self.min_size, 0.1, 0.05)

    # compute the normal of each Lidar point by KNN and PCA
    def comp_normal(self):
        nbrs = NearestNeighbors(n_neighbors=self.n_nbrs, algorithm='ball_tree').fit(self.lidar_pts)
        distances, indices = nbrs.kneighbors(self.lidar_pts)
        self.nbrs_indices = indices
        pca = PCA(n_components=3)
        for i in xrange(0, len(self.lidar_pts)):
            pca.fit(self.lidar_pts[indices[i]])
            # the last principle component is the z-axis of the plane
            self.lidar_normal.append(pca.components_[-1])

    # greedily growing regions from random seed
    def find_board(self):
        # all points
        pts_set = set(range(0, len(self.lidar_pts)))
        # greedily seeds until no point unassigned
        while len(pts_set) != 0:
            # choose random seed from the point set then remove it from the whole set
            pt_j = random.sample(pts_set, 1)[0]
            pts_set.discard(pt_j)
            # segment seeds from p_j
            board_set = set([pt_j])
            converge = False
            while not converge:
                converge = True
                for pt_i in pts_set:
                    # check if ones of pt_i's neighbors is in the board set
                    # and pt_i has a similar normal compared to the seed's
                    if self.check_neighbor(pt_i, board_set) and \
                       abs(self.lidar_normal[pt_i].dot(self.lidar_normal[pt_j])) > self.tau_seg:
                        converge = False
                        board_set.add(pt_i)
                pts_set -= board_set
            self.boards_set.append(board_set)

    # check if ones of pt_i's neighbors is in the set
    def check_neighbor(self, pt_i, pts_set):
        for i in self.nbrs_indices[pt_i]:
            if i in pts_set:
                return True
        return False

    # final filtering step removes segments
    # which are either smaller than a checkerboard or not planar enough
    # @params: min_size: minimum number of points for a valid board
    #          min_ratio: minimum ratio of explained variance by second principle component
    #                     in order to remove a single line of range points
    #          max_ratio: maximum ratio of explained variance by third principle component
    #                     in order to remove non planar boards
    def filter_board(self, min_size, min_ratio, max_ratio):
        print 'initial size: %d' % len(self.boards_set)
        # filter out planes with only a small number of points
        temp_boards_set = []
        for i in xrange(0, len(self.boards_set)):
            if len(self.boards_set[i]) >= min_size:
                temp_boards_set.append(self.boards_set[i])
        self.boards_set = temp_boards_set
        # filter out planes not planar enough
        temp_boards_set = []
        pca = PCA(n_components=3)
        for i in xrange(0, len(self.boards_set)):
            train_set = np.array([self.lidar_pts[x] for x in self.boards_set[i]])
            pca.fit(train_set)
            if pca.explained_variance_ratio_[1] > min_ratio and \
               pca.explained_variance_ratio_[2] < max_ratio:
                temp_boards_set.append(self.boards_set[i])
                self.boards_pts_set.append(train_set)
        self.boards_set = temp_boards_set



