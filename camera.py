from corner import Corner_Detection
from checkerboard import Board_Recovery
import numpy as np
import scipy.spatial as spatial
import math
import sys
import cv2

'''
Implement checkerboard pattern matching between different cameras and
calibration between cameras
'''

class Camera_Calibration:

    def __init__(self, img, corner_dist):
        # images from multiple cameras
        self.img = img
        # distance between adjacent corner
        self.corner_dist = corner_dist
        # corner detection result for each camera
        self.corners = []
        # checkerboard recovery result for each camera
        self.boards = []
        # pattern matching result for each camera
        self.matching = []
        # camera intrinsic matrix
        self.intrinsic = []
        # camera extrinsic matrix
        # ith translation/rotation is the transformation
        # from (i+1)th camera to the first camera (reference)
        self.rotation = []
        self.translation = []
        # camera distortion parameters
        self.distortion = []
        # camera rectification rotation
        self.rect_rot = []
        # camera projection matrix after rectification
        self.cam_proj = []
        # 3d points recovery for each board
        self.boards_pts_set = []
        # corner detection and pattern recovery
        for i in xrange(0, len(self.img)):
            print "Image %d:" % i
            corner = Corner_Detection(self.img[i], 0.02)
            corner.find_corners()
            board = Board_Recovery(corner.corners_pts, corner.corners_v1, corner.corners_v2)
            board.find_boards()
            self.corners.append(corner)
            self.boards.append(board)
        self.check_result()

    # check corner detection and checkerboard recovery results
    def check_result(self):
        for i in xrange(0, len(self.boards)):
            if len(self.boards[i].chessboards) < 3:
                print "The No. of recovered checkerboard in image %d is %d less than 3" \
                      % (i, len(self.boards[i].chessboards))
                sys.exit(1)

    # main function: match patterns between two cameras and then calibrate the two cameras
    def match_cali(self):
        self.match_pattern()
        self.camera_cali()
        self.recover_points()

    # match checkerboards between cameras taking first img as reference
    def match_pattern(self):
        for i in xrange(1, len(self.img)):
            print "match image %d to %d" % (i, 0)
            self.match_against_ref(self.boards[0].chessboards,
                                   self.corners[0].corners_pts,
                                   self.boards[i].chessboards,
                                   self.corners[i].corners_pts)
            # DEBUG: check number of boards
            print "reference image has %d boards" % len(self.boards[0].chessboards)
            print "target image has %d boards" % len(self.boards[i].chessboards)

    # calibrate camera given corners, patterns and pattern matching
    def camera_cali(self):
        # calibration a single camera
        '''
        for c in xrange(0, 1):
            # 3d points in real world space
            obj_pts = []
            # 2d points in image plane
            img_pts = []
            boards = self.boards[c].chessboards
            corners = self.corners[c].corners_pts
            # check if this image is the reference image
            if c == 0:
                for b in xrange(0, len(boards)):
                    h, w = len(boards[b]), len(boards[b][0])
                    obj_pts_board = np.zeros((h * w, 3), np.float32)
                    img_pts_board = np.zeros((h * w, 2), np.float32)
                    board = boards[b]
                    for i in xrange(0, h):
                        for j in xrange(0, w):
                            obj_pts_board[i * w + j, 0] = j * self.corner_dist
                            obj_pts_board[i * w + j, 1] = i * self.corner_dist
                            img_pts_board[i * w + j, 0] = corners[int(board[i, j])][0]
                            img_pts_board[i * w + j, 1] = corners[int(board[i, j])][1]
                    obj_pts.append(obj_pts_board)
                    img_pts.append(img_pts_board)

            camera_mat = np.zeros((3, 3), np.float32)
            camera_mat[0, 0] = 900.
            camera_mat[0, 2] = len(self.img[c][0]) / 2
            camera_mat[1, 1] = 900.
            camera_mat[1, 2] = len(self.img[c]) / 2
            camera_mat[2, 2] = 1.
            ret, mtx, dist, rv, tv = cv2.calibrateCamera(obj_pts, img_pts,
                                                         (len(self.img[c][0]), len(self.img[c])),
                                                         camera_mat, None, None, None,
                                                         flags=(cv2.CALIB_FIX_ASPECT_RATIO +
                                                                cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_K3))

        print(mtx, dist)

        '''
        # first camera/image is the reference camera/image
        boards_ref = self.boards[0].chessboards
        corners_ref = self.corners[0].corners_pts
        # only support two-camera calibration
        for c in xrange(1, len(self.boards)):
            match = self.matching[2 * (c - 1)]
            rot = self.matching[2 * (c - 1) + 1]
            print match
            print rot
            obj_pts_ref = []
            img_pts_ref = []
            obj_pts_tar = []
            img_pts_tar = []
            boards_tar = self.boards[c].chessboards
            corners_tar = self.corners[c].corners_pts
            for b in xrange(0, len(match)):
                if match[b] != -1:
                    # reference camera
                    obj_pts_board_ref = []
                    img_pts_board_ref = []
                    board_ref = boards_ref[b]

                    for i in xrange(0, len(board_ref)):
                        for j in xrange(0, len(board_ref[0])):
                            obj_pts_board_ref.append([j * self.corner_dist, i * self.corner_dist, 0])
                            img_pts_board_ref.append(corners_ref[board_ref[i, j]])

                    # target camera
                    obj_pts_board_tar = []
                    img_pts_board_tar = []
                    board_tar = boards_tar[int(match[b])]

                    for k in xrange(0, int(rot[b])):
                        board_tar = np.flipud(board_tar).T

                    for i in xrange(0, len(board_tar)):
                        for j in xrange(0, len(board_tar[0])):
                            obj_pts_board_tar.append([j * self.corner_dist, i * self.corner_dist, 0])
                            img_pts_board_tar.append(corners_tar[board_tar[i, j]])

                    obj_pts_ref.append(np.array(obj_pts_board_ref).astype(np.float32))
                    img_pts_ref.append(np.array(img_pts_board_ref).astype(np.float32))
                    obj_pts_tar.append(np.array(obj_pts_board_tar).astype(np.float32))
                    img_pts_tar.append(np.array(img_pts_board_tar).astype(np.float32))

            # visualize the corners found on image
            for no_board in xrange(0, len(img_pts_ref)):
                I_1 = self.img[0]
                I_2 = self.img[1]
                for i in xrange(0, len(img_pts_ref[no_board])):
                    x = int(math.ceil(img_pts_ref[no_board][i, 0]))
                    y = int(math.ceil(img_pts_ref[no_board][i, 1]))
                    I_1[y-3:y+3, x-3:x+3] = [0, 0, 255]
                cv2.imshow('left', I_1)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                for i in xrange(0, len(img_pts_tar[no_board])):
                    x = int(math.ceil(img_pts_tar[no_board][i, 0]))
                    y = int(math.ceil(img_pts_tar[no_board][i, 1]))
                    I_2[y-3:y+3, x-3:x+3] = [0, 0, 255]
                cv2.imshow('right', I_2)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # stereo camera calibration
            ret, cam_mat_1, cam_dist_1, cam_mat_2, cam_dist_2, R, T, E, F = \
                cv2.stereoCalibrate(obj_pts_ref, img_pts_ref, img_pts_tar, (len(self.img[c][0]), len(self.img[c])),
                                    flags=(cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4+cv2.CALIB_FIX_K5))
            rect_rot_1, rect_rot_2, cam_proj_1, cam_proj_2, Q, ROI1, ROI2 = \
                cv2.stereoRectify(cam_mat_1, cam_dist_1, cam_mat_2, cam_dist_2,
                                  (len(self.img[c][0]), len(self.img[c])), R, T,
                                  None, None, None, None, None, flags=(cv2.CALIB_ZERO_DISPARITY),
                                  alpha=0, newImageSize=(945, 702))
            self.intrinsic.append(cam_mat_1)
            self.intrinsic.append(cam_mat_2)
            self.distortion.append(cam_dist_1)
            self.distortion.append(cam_dist_2)
            self.rotation.append(R)
            self.translation.append(T)
            self.cam_proj.append(cam_proj_1)
            self.cam_proj.append(cam_proj_2)
            self.rect_rot.append(rect_rot_1)
            self.rect_rot.append(rect_rot_2)

    # find 3d points by triangulation after camera calibration
    def recover_points(self):
        # first camera/image is the reference camera/image
        boards_ref = self.boards[0].chessboards
        corners_ref = self.corners[0].corners_pts
        # only support two-camera triangulation for now
        for c in xrange(1, len(self.boards)):
            match = self.matching[2 * (c - 1)]
            rot = self.matching[2 * (c - 1) + 1]
            boards_tar = self.boards[c].chessboards
            corners_tar = self.corners[c].corners_pts
            for b in xrange(0, len(match)):
                if match[b] != -1:
                    # reference camera
                    img_pts_board_ref = []
                    board_ref = boards_ref[b]

                    for i in xrange(0, len(board_ref)):
                        for j in xrange(0, len(board_ref[0])):
                            img_pts_board_ref.append(corners_ref[board_ref[i, j]])

                    # target camera
                    img_pts_board_tar = []
                    board_tar = boards_tar[int(match[b])]

                    for k in xrange(0, int(rot[b])):
                        board_tar = np.flipud(board_tar).T

                    for i in xrange(0, len(board_tar)):
                        for j in xrange(0, len(board_tar[0])):
                            img_pts_board_tar.append(corners_tar[board_tar[i, j]])

                    # TODO: update corner pixel coordinates to new rectified coordinate and
                    # use rectified projection matrix
                    cam_proj_1 = np.concatenate((self.intrinsic[0], np.zeros((3, 1))), axis=1)
                    cam_proj_2 = self.intrinsic[1].dot(np.concatenate((self.rotation[0], self.translation[0]), axis=1))
                    obj_pts_board = cv2.triangulatePoints(cam_proj_1, cam_proj_2,
                                                          np.array(img_pts_board_ref).astype(np.float32).T,
                                                          np.array(img_pts_board_tar).astype(np.float32).T)
                    obj_pts_board = obj_pts_board.T
                    obj_pts_board = obj_pts_board[:, 0:2]/obj_pts_board[:, 2:]
                    self.boards_pts_set.append(obj_pts_board)

            # visualize the corners found on image
            '''
            I_1 = self.img[0]
            I_2 = self.img[1]
            no_board = 10
            for i in xrange(0, len(img_pts_ref[no_board])):
                x = int(math.ceil(img_pts_ref[no_board][i, 0]))
                y = int(math.ceil(img_pts_ref[no_board][i, 1]))
                I_1[y-2:y+2, x-2:x+2] = [255, 0, 0]
            cv2.imshow('left', I_1)
            cv2.waitKey(0)
            for i in xrange(0, len(img_pts_tar[no_board])):
                x = int(math.ceil(img_pts_tar[no_board][i, 0]))
                y = int(math.ceil(img_pts_tar[no_board][i, 1]))
                I_2[y-2:y+2, x-2:x+2] = [255, 0, 0]
            cv2.imshow('right', I_2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''

    # match target checkerboard against reference checkerboard
    def match_against_ref(self, boards_ref, corners_ref, boards_tar, corners_tar):
        corners_ref = np.array(corners_ref)
        corners_tar = np.array(corners_tar)

        # numbers of checkerboards
        n_ref = len(boards_ref)
        n_tar = len(boards_tar)

        # compute reference and target chessboard center
        means_ref = self.board_mean(boards_ref, corners_ref)
        means_tar = self.board_mean(boards_tar, corners_tar)

        # determine outlier corner re-projection error based on the maximum board center distance
        tau = 0.2 * spatial.distance.pdist(means_ref).max()

        # two checkerboards a group in one image to match another group of two boards in another image
        matchings = []
        for i_ref in xrange(0, n_ref):
            for j_ref in xrange(0, n_ref):
                for i_tar in xrange(0, n_tar):
                    for j_tar in xrange(0, n_tar):
                        # check if a group contains two same boards
                        if i_ref == j_ref or i_tar == j_tar:
                            continue

                        # compute 2d similarity transformation p_t * (scale * R) + T = p_r
                        # target point: 1x2
                        v1 = means_ref[j_ref] - means_ref[i_ref]
                        v2 = means_tar[j_tar] - means_tar[i_tar]
                        s = np.linalg.norm(v1) / np.linalg.norm(v2)
                        r = math.acos(min(max((v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))), -1), 1))
                        R = s * np.array([[math.cos(r), -1. * math.sin(r)], [math.sin(r), math.cos(r)]]).T
                        T = means_ref[i_ref] - means_tar[i_tar].dot(R)

                        # project target to reference board coordinates
                        means_tar_proj = means_tar.dot(R) + T

                        # greedily compute matching until hitting outlier threshold
                        dist = spatial.distance.squareform(spatial.distance.pdist(
                                                           np.concatenate((means_ref, means_tar_proj), axis=0)))
                        dist = dist[:len(means_ref), len(means_ref):]
                        matching = np.ones(len(means_ref)) * -1

                        while True:
                            val = dist.min()
                            if val > tau:
                                break
                            pos = np.argwhere(dist == val).flatten()
                            row = pos[0]
                            col = pos[1]

                            # check if the numbers of row and column match
                            if (len(boards_ref[row]) == len(boards_tar[col]) and
                               len(boards_ref[row][0]) == len(boards_tar[col][0])) or \
                               (len(boards_ref[row]) == len(boards_tar[col][0]) and
                               len(boards_ref[row][0]) == len(boards_tar[col])):
                                matching[row] = col
                                dist[row, :] = np.inf
                                dist[:, col] = np.inf
                            else:
                                dist[row, col] = np.inf

                        # need at least three matching
                        if np.sum(matching != -1) >= 3:
                            matchings.append(matching)

        # make matchings row vector unique and sort by number of non-zero entries
        matchings = np.array(matchings)
        matchings = np.vstack({tuple(row) for row in matchings})
        matchings = matchings[(matchings == -1).sum(axis=1).argsort()]

        # score each matching and find the best matching
        max_score = -float('inf')
        max_idx = -1
        max_rotation = []
        for i in xrange(0, len(matchings)):
            rotation, score = self.score_matching(matchings[i],
                                                  boards_ref, corners_ref, means_ref,
                                                  boards_tar, corners_tar, means_tar, tau)
            if score > max_score:
                max_score = score
                max_idx = i
                max_rotation = rotation

        # check if no valid matching found
        if max_idx != -1:
            self.matching.append(matchings[max_idx])
            self.matching.append(max_rotation)

    # compute given checkerboard center location
    @staticmethod
    def board_mean(boards, corners):
        means = np.zeros([len(boards), 2])
        for i in xrange(0, len(boards)):
            means[i, :] = np.mean(corners[boards[i].flatten()], axis=0)
        return means

    # score matching
    @staticmethod
    def score_matching(matching, boards_ref, corners_ref, means_ref, boards_tar, corners_tar, means_tar, tau):
        num_matched = np.sum(matching != -1)

        # compute affine transformation ref = tar*A + b by least squares fit to all matched boards
        H = np.zeros([num_matched * 2, 6])
        H[0:-1:2, 0:2] = means_tar[matching[matching != -1].astype(int)]
        H[0:-1:2, 4] = 1
        H[1:len(H):2, 2:4] = means_tar[matching[matching != -1].astype(int)]
        H[1:len(H):2, 5] = 1
        x = np.zeros([num_matched * 2, 1])
        x[0:-1:2] = means_ref[matching != -1][:, 0].reshape(1, -1).T
        x[1:len(H):2] = means_ref[matching != -1][:, 1].reshape(1, -1).T

        y = np.linalg.inv(H.T.dot(H)).dot(H.T).dot(x)
        y = y.flatten()
        A = np.array([[y[0], y[2]], [y[1], y[3]]])
        b = np.array([y[4], y[5]])

        score = 0
        rotation = np.zeros(len(matching))
        for i in xrange(0, len(matching)):
            j = int(matching[i])
            if j != -1:
                dist, rot = Camera_Calibration.min_corner_dist(boards_ref[i], boards_tar[j],
                                                               corners_ref, corners_tar, A, b)
                score = score - dist/tau + 1
                rotation[i] = rot

        return rotation, score

    # for all corners in all chessboards in the reference image
    # compute minimum corner distance to transformed target image
    @staticmethod
    def min_corner_dist(board_ref, board_tar, corners_ref, corners_tar, A, b):
        # initialize distance and rotation
        dist = float('inf')
        rot = 0

        # for all rotations do (i = 0 => no rotation) matching ambiguity
        for i in xrange(0, 4):
            p_ref = corners_ref[board_ref.flatten('F')]
            p_tar = corners_tar[board_tar.flatten('F')].dot(A) + b

            # check if target board and reference board have the same size
            if len(board_ref) == len(board_tar) and \
               len(board_ref[0]) == len(board_tar[0]):

                dist_ = p_ref - p_tar
                dist_ = np.mean(np.sqrt(np.sum(np.square(dist_), axis=1)))
                if dist_ < dist:
                    dist = dist_
                    rot = i
            board_tar = np.flipud(board_tar).T

        return dist, rot

