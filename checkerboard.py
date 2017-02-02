import numpy as np

'''
Implement checkerboard recovery for multiple checkerboards in one single image
with corners already detected
'''


class Board_Recovery:

    def __init__(self, corners_1, corners_1_v1, corners_1_v2):
        self.c1 = corners_1
        self.c1_v1 = corners_1_v1
        self.c1_v2 = corners_1_v2
        self.chessboards = []

    # Recover chessboard (main function)
    def find_boards(self):
        print "Recover checkerboard patterns..."
        # seed every corner
        for i in xrange(0, len(self.c1)):
            # initialize 3x3 chessboard from seed i
            chessboard = self.init_board(i).astype(int)

            # check if this is a valid initial guess
            if len(chessboard) == 0 or self.board_energy(chessboard) > 0:
                continue

            # expand one of the four chessboard borders by one col or row
            while True:
                # current board energy
                energy = self.board_energy(chessboard)
                # 4 expansion strategies and their energies
                proposal = []
                p_energy = []
                for j in xrange(0, 4):
                    proposal.append(self.grow_board(chessboard, j))
                    p_energy.append(self.board_energy(proposal[j]))

                min_idx = p_energy.index(min(p_energy))
                if p_energy[min_idx] < energy:
                    chessboard = proposal[min_idx]
                else:
                    break

            # if board energy is low enough (high quality pattern)
            if self.board_energy(chessboard) < -10:
                self.add_board(chessboard)

    # initialize 3x3 chessboard from seed idx
    def init_board(self, idx):
        # return if not enough corners to form 3x3 board
        if len(self.c1) < 9:
            return np.array([])
        # initial chessboard -1 represent no corner found yet
        chessboard = np.ones([3, 3]) * -1
        # extract corner orientation
        v1 = np.array(self.c1_v1[idx])
        v2 = np.array(self.c1_v2[idx])
        chessboard[1, 1] = idx
        dist1 = np.zeros(2)
        dist2 = np.zeros(6)
        # find left, right, top, bottom neighbors
        chessboard[1, 2], dist1[0] = self.dir_neighbor(idx, +1 * v1, chessboard)
        chessboard[1, 0], dist1[1] = self.dir_neighbor(idx, -1 * v1, chessboard)
        chessboard[2, 1], dist2[0] = self.dir_neighbor(idx, +1 * v2, chessboard)
        chessboard[0, 1], dist2[1] = self.dir_neighbor(idx, -1 * v2, chessboard)
        # find top-left/top-right/bottom-left/bottom-right neighbors
        chessboard[0, 0], dist2[2] = self.dir_neighbor(int(chessboard[1, 0]), -1 * v2, chessboard)
        chessboard[2, 0], dist2[3] = self.dir_neighbor(int(chessboard[1, 0]), +1 * v2, chessboard)
        chessboard[0, 2], dist2[4] = self.dir_neighbor(int(chessboard[1, 2]), -1 * v2, chessboard)
        chessboard[2, 2], dist2[5] = self.dir_neighbor(int(chessboard[1, 2]), +1 * v2, chessboard)

        if np.any(np.isinf(dist1)) or np.any(np.isinf(dist2)) or \
           np.std(dist1)/np.mean(dist1) > 0.3 or np.std(dist2)/np.mean(dist2) > 0.3:
            return np.array([])
        return chessboard

    # find the neighbor corner along the direction v
    def dir_neighbor(self, idx, v, chessboard):
        used = chessboard[chessboard != -1]
        unused = [i for i in xrange(0, len(self.c1)) if i not in used]
        # direction and distance to unused corners
        dir = np.array(self.c1)[unused] - np.ones([len(unused), 1]) * (np.array(self.c1[idx]).reshape(1, -1))
        dist = (dir[:, 0] * v[0] + dir[:, 1] * v[1]).reshape(1, -1).T

        # distances
        v = v.reshape(1, -1)
        dist_edge = dir - dist.dot(v)
        dist_edge = np.sqrt(np.sum(np.square(dist_edge), axis=1).reshape(1, -1).T)
        dist_point = dist
        dist_point[dist_point < 0] = float("inf")

        # find best neighbor
        min_idx = np.argmin(dist_point + 5 * dist_edge)
        min_dist = dist_point[min_idx] + 5 * dist_edge[min_idx]

        return unused[min_idx], min_dist

    # compute given chessboard energy for structure recovery
    def board_energy(self, chessboard):
        corners = np.array(self.c1)
        # energy: number of corners
        E_corners = -1. * len(chessboard) * len(chessboard[0])

        # energy: structure
        E_structure = 0

        # for each rows find triples
        for j in xrange(0, len(chessboard)):
            for k in xrange(0, len(chessboard[0]) - 2):
                x = corners[chessboard[j, k: k+3]]
                E_structure = max(E_structure, np.linalg.norm(x[0, :] + x[2, :] - 2 * x[1, :]) /
                                  np.linalg.norm(x[0, :] - x[2, :]))
        # for each column find triples
        for j in xrange(0, len(chessboard[0])):
            for k in xrange(0, len(chessboard) - 2):
                x = corners[chessboard[k: k+3, j], :]
                E_structure = max(E_structure, np.linalg.norm(x[0, :] + x[2, :] - 2 * x[1, :]) /
                                  np.linalg.norm(x[0, :] - x[2, :]))
        return E_corners - E_corners * E_structure

    # expand current chessboard border
    def grow_board(self, chessboard, border_type):
        # check if chessboard is empty
        if len(chessboard) == 0:
            return chessboard

        # extract corners pixel location
        corners = np.array(self.c1)

        # list of unused corner index
        used = chessboard[chessboard != -1]
        unused = [i for i in xrange(0, len(corners)) if i not in used]

        # candidates from unused corners
        cand = corners[unused]

        # four types of expansion
        # 0 right; 1 bottom ; 2 left; 3 top
        if border_type == 0:
            # use 3 rightmost columns
            pred = Board_Recovery.predict_corners(corners[chessboard[:, -3]],
                                                  corners[chessboard[:, -2]],
                                                  corners[chessboard[:, -1]])
            idx = Board_Recovery.assign_closest_corners(cand, pred)
            if len(idx) != 0:
                idx = np.array(unused)[idx].reshape(1, -1)
                expand_board = np.concatenate((chessboard, idx.T), axis=1)
                return expand_board
        elif border_type == 1:
            # use 3 bottom rows
            pred = Board_Recovery.predict_corners(corners[chessboard[-3, :]],
                                                  corners[chessboard[-2, :]],
                                                  corners[chessboard[-1, :]])
            idx = Board_Recovery.assign_closest_corners(cand, pred)
            if len(idx) != 0:
                idx = np.array(unused)[idx].reshape(1, -1)
                expand_board = np.concatenate((chessboard, idx), axis=0)
                return expand_board

        elif border_type == 2:
            # use 3 leftmost columns
            pred = Board_Recovery.predict_corners(corners[chessboard[:, 2]],
                                                  corners[chessboard[:, 1]],
                                                  corners[chessboard[:, 0]])
            idx = Board_Recovery.assign_closest_corners(cand, pred)
            if len(idx) != 0:
                idx = np.array(unused)[idx].reshape(1, -1)
                expand_board = np.concatenate((idx.T, chessboard), axis=1)
                return expand_board
        elif border_type == 3:
            # use 3 top rows
            pred = Board_Recovery.predict_corners(corners[chessboard[2, :]],
                                                  corners[chessboard[1, :]],
                                                  corners[chessboard[0, :]])
            idx = Board_Recovery.assign_closest_corners(cand, pred)
            if len(idx) != 0:
                idx = np.array(unused)[idx].reshape(1, -1)
                expand_board = np.concatenate((idx, chessboard), axis=0)
                return expand_board
        return chessboard

    # add chessboard (case 1: no overlapping; case 2: overlapping but with lower energy)
    def add_board(self, chessboard):
        # check if new chessboard proposal overlaps with existing chessboard
        overlap = np.zeros([len(self.chessboards), 2])
        for j in xrange(0, len(self.chessboards)):
            cur_board = self.chessboards[j].flatten()
            for k in xrange(0, len(cur_board)):
                if np.any(chessboard == cur_board[k]):
                    overlap[j, 0] = 1
                    overlap[j, 1] = self.board_energy(self.chessboards[j])
                    break
        # add chessboard (case 1: no overlapping; case 2: overlapping but with lower energy)
        if not any(overlap[:, 0] == 1):
            self.chessboards.append(chessboard)
        else:
            idx = np.argwhere(overlap[:, 0] == 1)
            if not np.any(overlap[idx, 1] <= self.board_energy(chessboard)):
                idx = idx.flatten().tolist()
                chessboard_temp = [self.chessboards[i] for i in xrange(0, len(self.chessboards))
                                   if i not in idx]
                self.chessboards = chessboard_temp

    # given border cols or rows predict next col or row location
    @staticmethod
    def predict_corners(p1, p2, p3):
        # compute vectors
        v1 = p2 - p1
        v2 = p3 - p2
        # predict angles
        a1 = np.arctan2(v1[:, 1], v1[:, 0])
        a2 = np.arctan2(v2[:, 1], v1[:, 0])
        a3 = (2 * a2 - a1).reshape(1, -1).T
        # predict scales
        s1 = np.sqrt(np.sum(np.square(v1), axis=1))
        s2 = np.sqrt(np.sum(np.square(v2), axis=1))
        s3 = (2. * s2 - s1).reshape(1, -1).T
        predict = p3 + 0.75 * np.multiply(s3.dot(np.ones([1, 2])),
                                          np.concatenate((np.cos(a3), np.sin(a3)), axis=1))
        return predict

    # given predicted row or col location find closest corners
    @staticmethod
    def assign_closest_corners(cand, pred):
        # return 0 if not enough candidates
        if len(cand) < len(pred):
            return np.array([])

        # build distance matrix
        D = np.zeros([len(cand), len(pred)])
        for i in xrange(0, len(pred)):
            delta = cand - pred[i, :]
            D[:, i] = np.sqrt(np.sum(np.square(delta), axis=1))

        idx = np.zeros(len(pred))
        # greed search for the closest corner for each pred
        for i in xrange(0, len(pred)):
            pos = np.argwhere(D == D.min())
            idx[pos[0, 1]] = pos[0, 0]
            D[pos[0, 0]] = np.inf
            D[:, pos[0, 1]] = np.inf

        return idx.astype(int)



