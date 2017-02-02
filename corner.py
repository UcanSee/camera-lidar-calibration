import cv2
import math
import numpy as np
from scipy import signal
from operator import itemgetter

'''
Implement corner detection for multiple checkerboards in one single image
'''

class Corner_Detection:
    # Parameters for creating corner patch for likelihood calculation
    radius = np.array([4, 8, 12])
    template_props = np.array([[0, math.pi/2, radius[0]], [math.pi/4, -math.pi/4, radius[0]],
                               [0, math.pi/2, radius[1]], [math.pi/4, -math.pi/4, radius[1]],
                               [0, math.pi/2, radius[2]], [math.pi/4, -math.pi/4, radius[2]]])
    # Parameters for gradient calculation
    du = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    def __init__(self, img, tau):
        # convert image to gray scale
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # scale image between 0 and 1
        img_min = self.img.min()
        img_max = self.img.max()
        # grayscale image
        self.img = 1.0 * (self.img - img_min) / (img_max - img_min)
        # corner likelihood
        self.img_corners = np.zeros(self.img.shape)
        # corner points u, v
        self.corners_pts = []
        # corner score threshold
        self.tau = tau

    # Find corners (main function)
    def find_corners(self):
        print "Find corners..."
        # 1. calculate corner likelihood value for each pixel
        self.cal_likelihood()
        # 2. find corner candidates
        self.non_max_sup(3, 0.025, 5)
        # 3. calculate gradient and edge orientation
        self.refine_corners(10)  # radius = 10 pixels
        # 4. scoring
        self.score_corners()
        # 5. remove low scoring corners and refine
        self.score_threshold()

    # Calculate the likelihood of the image to corner patches
    def cal_likelihood(self):
        for t in range(0, len(self.template_props)):
            template = Corner_Detection.create_patch(self.template_props[t, 0],
                                                     self.template_props[t, 1],
                                                     self.template_props[t, 2])

            img_corners_a1 = signal.convolve2d(self.img, template[:, :, 0], 'same')
            img_corners_a2 = signal.convolve2d(self.img, template[:, :, 1], 'same')
            img_corners_b1 = signal.convolve2d(self.img, template[:, :, 2], 'same')
            img_corners_b2 = signal.convolve2d(self.img, template[:, :, 3], 'same')

            # compute mean
            img_corners_mu = (img_corners_a1 + img_corners_a2 + img_corners_b1 + img_corners_b2) / 4

            # case 1: a = white, b = black
            img_corners_a = np.minimum(img_corners_a1 - img_corners_mu, img_corners_a2 - img_corners_mu)
            img_corners_b = np.minimum(img_corners_mu - img_corners_b1, img_corners_mu - img_corners_b2)
            img_corners_1 = np.minimum(img_corners_a, img_corners_b)

            # case 2: b = white, a = black
            img_corners_a = np.minimum(img_corners_mu - img_corners_a1, img_corners_mu - img_corners_a2)
            img_corners_b = np.minimum(img_corners_b1 - img_corners_mu, img_corners_b2 - img_corners_mu)
            img_corners_2 = np.minimum(img_corners_a, img_corners_b)

            # update corner map
            self.img_corners = np.maximum(self.img_corners, img_corners_1)
            self.img_corners = np.maximum(self.img_corners, img_corners_2)

    # Non maximum suppression of the likelihood map
    def non_max_sup(self, n, tau, margin):
        width = int(len(self.img[0]))
        height = int(len(self.img))

        for i in xrange(n + margin, width - n - margin, n + 1):
            for j in xrange(n + margin, height - n - margin, n + 1):
                max_i = i
                max_j = j
                max_val = self.img_corners[j, i]

                for i_2 in xrange(i, i + n + 1):
                    for j_2 in xrange(j, j + n + 1):
                        curr_val = self.img_corners[j_2, i_2]
                        if curr_val > max_val:
                            max_i = i_2
                            max_j = j_2
                            max_val = curr_val

                failed = False
                for i_2 in xrange(max_i - n, min(max_i + n, width - margin) + 1):
                    for j_2 in xrange(max_j - n, min(max_j + n, height - margin) + 1):
                        curr_val = self.img_corners[j_2, i_2]
                        if curr_val > max_val and (i_2 < i or i_2 > i + n or j_2 < j or j_2 > j + n):
                            failed = True
                            break
                    if failed:
                        break
                if max_val >= tau and not failed:
                    self.corners_pts.append([max_i, max_j])

    # Refine corners by gradient statistics
    def refine_corners(self, r):
        # Compute gradient and edge orientation
        img_du = signal.convolve2d(self.img, self.du, 'same')
        img_dv = signal.convolve2d(self.img, self.du.T, 'same')
        img_du[abs(img_du) < 1e-10] = 0 # arctan2(0, 0) = 0; arctan2(1e-15, 0) = pi/2
        img_dv[abs(img_dv) < 1e-10] = 0
        img_angle = np.arctan2(img_dv, img_du)
        img_weight = np.sqrt(np.square(img_du) + np.square(img_dv))
        # let angle between 0 and pi
        img_angle[img_angle < 0] = img_angle[img_angle < 0] + math.pi
        img_angle[img_angle > math.pi] = img_angle[img_angle > math.pi] - math.pi

        width = int(len(self.img[0]))
        height = int(len(self.img))

        # Initialize two edge orientations for corners (e.g. |__ 0 and 90 degree)
        self.corners_v1 = [[0, 0] for l in range(0, len(self.corners_pts))]
        self.corners_v2 = [[0, 0] for l in range(0, len(self.corners_pts))]

        for i in xrange(0, len(self.corners_pts)):
            cu = self.corners_pts[i][0]
            cv = self.corners_pts[i][1]
            img_angle_sub = img_angle[max(cv - r, 0): min(cv + r + 1, height),
                                      max(cu - r, 0): min(cu + r + 1, width)]
            img_weight_sub = img_weight[max(cv - r, 0): min(cv + r + 1, height),
                                        max(cu - r, 0): min(cu + r + 1, width)]
            # Obtain initial orientations by gradient statistics and mean shift
            v1, v2 = self.edge_ori(img_angle_sub, img_weight_sub)
            self.corners_v1[i] = v1
            self.corners_v2[i] = v2

            # if invalid edge orientations
            if (abs(v1[0]) < 1e-10 and abs(v1[1]) < 1e-10) or (abs(v2[0]) < 1e-10 and abs(v2[1]) < 1e-10):
                continue

            v1 = np.array(v1)
            v2 = np.array(v2)

            # corner orientation refinement
            A1 = np.zeros([2, 2])
            A2 = np.zeros([2, 2])
            # assemble matrix
            for u in xrange(max(cu - r, 0), min(cu + r + 1, width)):
                for v in xrange(max(cv - r, 0), min(cv + r + 1, height)):
                    # pixel gradient orientation
                    p_ori = np.array([img_du[v, u], img_dv[v, u]])
                    p_norm = np.linalg.norm(p_ori)
                    if p_norm < 0.1:
                        continue
                    p_ori_norm = p_ori / p_norm

                    # pixel on the edge which orientation is v1
                    if abs(p_ori_norm.dot(v1.T)) < 0.25:
                        A1[0] += img_du[v, u] * p_ori
                        A1[1] += img_dv[v, u] * p_ori

                    if abs(p_ori_norm.dot(v2.T)) < 0.25:
                        A2[0] += img_du[v, u] * p_ori
                        A2[1] += img_dv[v, u] * p_ori

            eigval_1, eigvec_1 = np.linalg.eig(A1)
            eigval_2, eigvec_2 = np.linalg.eig(A2)
            v1 = eigvec_1[:, np.argmin(eigval_1)]
            v2 = eigvec_2[:, np.argmin(eigval_2)]
            self.corners_v1[i] = [v1[0], v1[1]]
            self.corners_v2[i] = [v2[0], v2[1]]

            v1 = v1.reshape(1, -1)  # for transpose otherwise always 1d vector
            v2 = v2.reshape(1, -1)

            # corner location refinement
            G = np.zeros([2, 2])
            b = np.zeros([2, 1])
            # assemble matrix
            for u in xrange(max(cu - r, 0), min(cu + r + 1, width)):
                for v in xrange(max(cv - r, 0), min(cv + r + 1, height)):
                    p_ori = np.array([img_du[v, u], img_dv[v, u]])
                    p_norm = np.linalg.norm(p_ori)
                    if p_norm < 0.1:
                        continue
                    p_ori_norm = p_ori / p_norm
                    p_ori = p_ori.reshape(1, -1)

                    # sub-pixel corner estimation
                    if u != cu or v != cv:
                        # relative position of pixel and distance to vectors
                        w = np.array([u - cu, v - cv])
                        d1 = np.linalg.norm(w - w.dot(v1.T.dot(v1)))
                        d2 = np.linalg.norm(w - w.dot(v2.T.dot(v2)))
                        if (d1 < 3 and abs(p_ori_norm.dot(v1.T)) < 0.25) or \
                           (d2 < 3 and abs(p_ori_norm.dot(v2.T)) < 0.25):
                            H = p_ori.T.dot(p_ori)
                            G += H
                            b += H.dot(np.array([[u], [v]]))

            # set new corner location if G has full rank
            if np.linalg.matrix_rank(G) == 2:
                corner_pos_old = np.array([cu, cv])
                corner_pos_new = np.linalg.solve(G, b).T.flatten()
                self.corners_pts[i] = [corner_pos_new[0], corner_pos_new[1]]
                # set corner to invalid if position update is very large
                if np.linalg.norm(corner_pos_new - corner_pos_old) >= 4:
                    self.corners_v1[i] = [0, 0]
                    self.corners_v2[i] = [0, 0]
            else:
                self.corners_v1[i] = [0, 0]
                self.corners_v2[i] = [0, 0]

        # remove corners without edges
        rm_idx = []
        for i in xrange(0, len(self.corners_pts)):
            if abs(self.corners_v1[i][0]) < 1e-10 and abs(self.corners_v1[i][1]) < 1e-10:
                rm_idx.append(i)
        corner_pos_new = [self.corners_pts[i] for i in xrange(0, len(self.corners_pts)) if i not in rm_idx]
        corner_v1_new = [self.corners_v1[i] for i in xrange(0, len(self.corners_pts)) if i not in rm_idx]
        corner_v2_new = [self.corners_v2[i] for i in xrange(0, len(self.corners_pts)) if i not in rm_idx]

        self.corners_pts = corner_pos_new
        self.corners_v1 = corner_v1_new
        self.corners_v2 = corner_v2_new
        self.img_weight = img_weight

    # Score each corners based on likelihood and orientation
    def score_corners(self):
        width = int(len(self.img[0]))
        height = int(len(self.img))
        self.corners_score = []
        for i in xrange(0, len(self.corners_pts)):
            # corner location
            u = int(round(self.corners_pts[i][0]))
            v = int(round(self.corners_pts[i][1]))
            score = []
            for j in xrange(0, len(self.radius)):
                score.append(0)
                r = self.radius[j]
                if r <= u < width - r and r <= v < height - r:
                    img_sub = self.img[v - r: v + r + 1, u - r: u + r + 1]
                    img_weight_sub = self.img_weight[v - r: v + r + 1, u - r: u + r + 1]
                    score[j] = Corner_Detection.corr_score(img_sub, img_weight_sub,
                                                           self.corners_v1[i], self.corners_v2[i])
            self.corners_score.append(max(score))

    # Remove low scoring corners and unify coordinate systems
    def score_threshold(self):
        # remove low scoring corners
        rm_idx = []
        for i in xrange(0, len(self.corners_pts)):
            if self.corners_score[i] < self.tau:
                rm_idx.append(i)
        corner_pos_new = [self.corners_pts[i] for i in xrange(0, len(self.corners_pts)) if i not in rm_idx]
        corner_v1_new = [self.corners_v1[i] for i in xrange(0, len(self.corners_pts)) if i not in rm_idx]
        corner_v2_new = [self.corners_v2[i] for i in xrange(0, len(self.corners_pts)) if i not in rm_idx]
        corner_score_new = [self.corners_score[i] for i in xrange(0, len(self.corners_pts)) if i not in rm_idx]

        self.corners_pts = corner_pos_new
        self.corners_v1 = corner_v1_new
        self.corners_v2 = corner_v2_new
        self.corners_score = corner_score_new

        # make v1[:][0] + v1[:][1] positive
        for i in xrange(0, len(self.corners_pts)):
            if sum(self.corners_v1[i]) < 0:
                self.corners_v1[i][0] *= -1
                self.corners_v1[i][1] *= -1

        # make all coordinate systems right-handed (reduces matching ambiguities from 8 to 4)
        corners_n1 = np.array(self.corners_v1)[:, np.array([1, 0])]
        corners_v2 = np.array(self.corners_v2)
        corners_n1[:, 1] *= -1
        flip = -1 * np.sign(np.multiply(corners_n1[:, 0], corners_v2[:, 0]) +
                            np.multiply(corners_n1[:, 1], corners_v2[:, 1]))
        flip = flip.reshape(1, -1).T
        corners_v2 = np.multiply(corners_v2, (flip.dot(np.ones([1, 2]))))
        self.corners_v2 = corners_v2.tolist()

    # Create patch for likelihood convolution
    @staticmethod
    def create_patch(angle_1, angle_2, radius):
        width = int(radius * 2 + 1)
        height = int(radius * 2 + 1)
        mu = radius
        mv = radius
        n1 = np.array([-math.sin(angle_1), math.cos(angle_1)])
        n2 = np.array([-math.sin(angle_2), math.cos(angle_2)])
        template = np.zeros([height, width, 4])
        for u in xrange(0, width):
            for v in xrange(0, height):
                vec = np.array([u - mu, v - mv])
                dist = np.linalg.norm(vec)

                s1 = vec.dot(n1.T)
                s2 = vec.dot(n2.T)
                if s1 <= -0.1 and s2 <= -0.1:
                    template[v, u, 0] = Corner_Detection.normpdf(dist, radius / 2)
                elif s1 >= 0.1 and s2 >= 0.1:
                    template[v, u, 1] = Corner_Detection.normpdf(dist, radius / 2)
                elif s1 <= -0.1 and s2 >= 0.1:
                    template[v, u, 2] = Corner_Detection.normpdf(dist, radius / 2)
                elif s1 >= 0.1 and s2 <= -0.1:
                    template[v, u, 3] = Corner_Detection.normpdf(dist, radius / 2)
        for t in xrange(0, 4):
                if abs(np.sum(template[:, :, t])) < 1e-10:
                    template[:, :, t] = float('nan')
                else:
                    template[:, :, t] = template[:, :, t] / np.sum(template[:, :, t])
        return template

    # Obtain probability from normal distribution with mean at 0
    @staticmethod
    def normpdf(x, sigma):
        return 1.0 / (sigma * (math.sqrt(2.0 * math.pi))) * math.pow(math.e, -x ** 2 / (2.0 * sigma ** 2))

    # Obtain two edge orientations for corners by computing a weighted orientation histogram
    @staticmethod
    def edge_ori(img_angle, img_weight):
        # Two orientations [cos(a) sin(a)]
        v1 = [0, 0]
        v2 = [0, 0]

        # Num of bins for histogram
        bin_size = 32

        vec_angle = img_angle.flatten('F')
        vec_weight = img_weight.flatten('F')

        # Convert edge normals to directions
        vec_angle += math.pi / 2.0
        vec_angle[vec_angle > math.pi] = vec_angle[vec_angle > math.pi] - math.pi

        # Create histogram
        angle_hist = np.zeros(bin_size)
        for i in xrange(0, len(vec_angle)):
            bin_num = int(max(min(math.floor(vec_angle[i] / (math.pi / bin_size)), bin_size - 1), 0))
            angle_hist[bin_num] += vec_weight[i]

        # Find modes (evident orientation) of smoothed histogram
        modes, angle_hist_smoothed = Corner_Detection.find_modes(angle_hist, 1)

        # if only one or no mode is found
        if len(modes) <= 1:
            return v1, v2

        # extract two strongest modes
        modes = modes[0:2]

        # compute corresponding orientation
        for i in xrange(0, len(modes)):
            modes[i].append(modes[i][0] * math.pi / bin_size)

        # sort modes by orientation
        modes = sorted(modes, key=itemgetter(2), reverse=False)

        # check if the difference between the two modes is two small
        delta_angle = min(modes[1][2] - modes[0][2], modes[0][2] + math.pi - modes[1][2])
        if delta_angle <= 0.3:
            return v1, v2

        # return the two orientations found
        v1 = [math.cos(modes[0][2]), math.sin(modes[0][2])]
        v2 = [math.cos(modes[1][2]), math.sin(modes[1][2])]
        return v1, v2

    # Obtain modes of smoothed histogram by mean shift
    @staticmethod
    def find_modes(hist, sigma):
        l = len(hist)

        # Create smoothed histogram
        hist_smoothed = np.zeros(l)
        # smooth parameters
        j = xrange(int(-round(2.0 * sigma)), int(round(2.0 * sigma) + 1))
        norm_pdf = np.array([Corner_Detection.normpdf(j_, sigma) for j_ in j])
        for i in xrange(0, l):
            idx = [(i + j_ ) % l for j_ in j]
            hist_smoothed[i] = np.sum(np.multiply(hist[idx], norm_pdf))

        modes = []

        # check if at least one entry is non-zero
        if all(abs(hist_smoothed - hist_smoothed[0]) < 1e-5):
            return modes

        # find modes
        for i in xrange(0, l):
            j = i
            while True:
                h0 = hist_smoothed[j]
                j1 = (j + 1) % l
                j2 = (j - 1) % l
                h1 = hist_smoothed[j1]
                h2 = hist_smoothed[j2]
                if h1 >= h0 and h1 >= h2:
                    j = j1
                elif h2 > h0 and h2 > h1:
                    j = j2
                else:
                    break
            if len(modes) == 0 or all([row[0] != j for row in modes]):
                modes.append([j, hist_smoothed[j]])

        # Sort modes according to weight
        modes = sorted(modes, key=itemgetter(1), reverse=True)
        return modes, hist_smoothed

    # Score corners by correlation
    @staticmethod
    def corr_score(img, img_weight, v1, v2):
        v1 = np.array(v1).reshape(1, -1)
        v2 = np.array(v2).reshape(1, -1)
        # center
        width = int(len(img_weight[0]))
        height = int(len(img_weight))
        c = np.ones([1, 2]) * (height - 1) / 2
        # gradient filter kernel (bandwidth = 3 px)
        img_filter = np.ones(img_weight.shape) * -1
        for x in xrange(0, width):
            for y in xrange(0, height):
                p1 = np.array([[x, y]]) - c
                p2 = p1.dot(v1.T.dot(v1))
                p3 = p1.dot(v2.T.dot(v2))
                if np.linalg.norm(p1 - p2) <= 1.5 or np.linalg.norm(p1 - p3) <= 1.5:
                    img_filter[y, x] = 1

        vec_weight = img_weight.flatten('F')
        vec_filter = img_filter.flatten('F')
        # normalize weight and filter
        vec_weight = (vec_weight - np.mean(vec_weight)) / np.std(vec_weight, ddof=1)
        vec_filter = (vec_filter - np.mean(vec_filter)) / np.std(vec_filter, ddof=1)
        # compute gradient score
        score_gradient = max(np.sum(np.multiply(vec_weight, vec_filter)) / (len(vec_weight) - 1), 0)
        # create intensity filter kernel
        template = Corner_Detection.create_patch(math.atan2(v1[0, 1], v1[0, 0]),
                                                 math.atan2(v2[0, 1], v2[0, 0]),
                                                 c[0, 0])
        # checkerboard responses
        a1 = np.sum(np.multiply(template[:, :, 0], img))
        a2 = np.sum(np.multiply(template[:, :, 1], img))
        b1 = np.sum(np.multiply(template[:, :, 2], img))
        b2 = np.sum(np.multiply(template[:, :, 3], img))
        mu = (a1 + a2 + b1 + b2) / 4.0

        # case 1: a = white, b = black
        score_a = min(a1 - mu, a2 - mu)
        score_b = min(mu - b1, mu - b2)
        score_1 = min(score_a, score_b)

        # case 2: a = black, b = white
        score_a = min(mu - a1, mu - a2)
        score_b = min(b1 - mu, b2 - mu)
        score_2 = min(score_a, score_b)

        score_intensity = max(max(score_1, score_2), 0)

        if math.isnan(score_intensity):
            score_intensity = 0.0

        # final score: product of gradient and intensity score
        score = score_gradient * score_intensity
        return score
