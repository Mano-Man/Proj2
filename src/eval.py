import numpy as np
from math import sqrt, floor
import sys
sys.path.append("../utils/")
sys.path.append("../../utils/")
from general_util import assertEqual

# ----------------------------------------------------------------------------------------------------------------------
#                                               Evaluation Metrics
# ----------------------------------------------------------------------------------------------------------------------

def zeroOneAccuracy(originalPermMat, proposedPermMat):
    """
    :param originalPermMat: np.ndarray of any dim: The original permutation or permutations: n_perms x t^2 indices
    :param proposedPermMat: np.ndarray of any dim: The proposed permutation or permutations: n_perms x t^2 indices
    :return:  Counts the amount of permutations we got right
    """
    if(originalPermMat.ndim == 1):
        return np.array_equal(originalPermMat,proposedPermMat)
    else:
        n_perms = originalPermMat.shape[0]
        return np.count_nonzero(np.logical_not((originalPermMat != proposedPermMat).sum(axis=1)))/n_perms


def directAccuracy(originalPermMat, proposedPermMat):
    """
    :param originalPermMat: np.ndarray of any dim: The original permutation or permutations
    :param proposedPermMat: np.ndarray of any dim: The proposed permutation or permutations
    :return:  Returns the 1/0 average score over all the spots we got the permutation right
    """
    return np.count_nonzero(originalPermMat == proposedPermMat) * 1.0 / originalPermMat.size


def neighborAccuracy(originalPermMat, proposedPermMat, dims=None):
    """
    :param originalPermMat: Target Permutation matrix
            Opt1: 2D np.ndarray of n_perms x t_squared
            Opt2: 2D np.ndarray of n_perms x m x n (Allow of different horiz/vertic cuts).
                    +
    :param proposedPermMat: The proposed permutation (expects 2D np.array) arranged in a grid view
    :return: The amount of immediate neighbors we
    """

    n_rows, n_cols, n_perms = _determineDimensionality(originalPermMat, dims)
    totals = 0

    # Also possible to use the following, instead of iteratively adding to register:
    # bulk = 2*(n_cols-1)*(n_rows -1)
    # edges = n_rows -1 + n_cols -1
    # total_neighbors = bulk + edges

    for i in range(n_perms):
        total_neighbors = num_correct = 0
        if n_perms == 1 and isinstance(originalPermMat[0], int): #TODO make this less ugly
            curr_target_perm = originalPermMat
            curr_proposed_perm = proposedPermMat
        else:
            curr_target_perm = originalPermMat[i]
            curr_proposed_perm = proposedPermMat[i]

        # Total amount of correct neighbors:
        for index in range(curr_target_perm.size):

            piece_num = curr_target_perm[index]
            next_ind = index + 1
            prop_index = np.where(curr_proposed_perm == piece_num)[0][0]
            if (next_ind % n_cols != 0):
                prop_right = prop_index + 1
                num_correct += 1 if (prop_right % n_cols != 0) and \
                                    curr_proposed_perm[prop_right] == curr_target_perm[next_ind]  else 0
                total_neighbors += 1

            next_ind = index + n_cols
            if (next_ind < len(curr_target_perm)):
                prop_down = prop_index + n_cols
                num_correct += 1 if (prop_down < len(curr_target_perm)) and \
                                    curr_proposed_perm[prop_down] == curr_target_perm[next_ind] else 0
                total_neighbors += 1
        totals += num_correct / (total_neighbors)

    return totals / n_perms


# ----------------------------------------------------------------------------------------------------------------------
#                                               Private Subroutines
# ----------------------------------------------------------------------------------------------------------------------

def _determineDimensionality(mat, dims=None):
    if dims is None:
        # Presume 2D np.array of n_perms x t^2
        if mat.ndim ==1 :
            n_perms = 1
            t_squared = mat.size
        else :
            n_perms, t_squared = mat.shape

        n_cols = n_rows = sqrt(t_squared)
        assert (floor(n_cols) == n_cols)  # Assert is integer
        n_cols = n_rows = int(n_cols)
    else:
        try:
            n_rows, n_cols, n_perms = dims
        except:
            n_rows, n_cols = dims
            n_perms = 1

    return n_rows, n_cols, n_perms


# ----------------------------------------------------------------------------------------------------------------------
#                                               Model Test Suite ``
# ----------------------------------------------------------------------------------------------------------------------


def unit_test():

    # Test Zero One Accuracy
    # Test directAccuracy
    assertEqual(zeroOneAccuracy(np.array([1, 2, 3, 4]), np.array([1, 3, 2, 4])), 0)
    assertEqual(zeroOneAccuracy(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])), 1)
    assertEqual(zeroOneAccuracy(np.array([4, 2, 3, 1]), np.array([1, 4, 2, 3])), 0)
    assertEqual(zeroOneAccuracy(np.array([[80, 75, 85, 90], [75, 80, 75, 85], [80, 80, 80, 90]]),
                               np.array([[79, 75, 85, 90], [75, 80, 75, 85], [80, 80, 80, 90]])), 2/3)

    # Test directAccuracy
    assertEqual(directAccuracy(np.array([1, 2, 3, 4]), np.array([1, 3, 2, 4])), 0.5)
    assertEqual(directAccuracy(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])), 1)
    assertEqual(directAccuracy(np.array([4, 2, 3, 1]), np.array([1, 4, 2, 3])), 0)
    assertEqual(directAccuracy(np.array([[80, 75, 85, 90], [75, 80, 75, 85], [80, 80, 80, 90]]),
                               np.array([[79, 75, 85, 90], [75, 80, 75, 85], [80, 80, 80, 90]])), 0.91666666666666666)

    # Test neighborAccuracy
    assertEqual(neighborAccuracy(np.array(range(9)), np.array([0, 1, 4, 3, 2, 5, 6, 7, 8])), 0.5)
    assertEqual(neighborAccuracy(np.array(range(9)), np.array([0, 1, 4, 3, 2, 5, 6, 7, 8]), (3, 3)), 0.5)
    # Assessment:
    # 0 1 4  0 1 2
    # 3 2 5  3 4 5
    # 6 7 8  6 7 8
    # Total Neighbors: 12 neighbors: (2*n - 1)*(m-1) + n-1= (2n-1)m -n
    # 6 Correct Spots: 0->1,0->3,3->6,5->8,6->7, 7->8

    assertEqual(neighborAccuracy(np.array(range(9)), np.array([8, 6, 7, 3, 4, 5, 1, 2, 0]), (3, 3)), 1 / 3)
    # Assessment:
    # 0 1 2  8 6 7
    # 3 4 5  3 4 5
    # 6 7 8  1 2 0
    # 4 Correct Spots: 1->2, 3->4, 4->5, 6->7
    assertEqual(neighborAccuracy(np.array(range(9)), np.array(range(9)), (3, 3)), 1)

    mat1 = np.tile(range(9), (9, 1))
    assertEqual(neighborAccuracy(mat1, mat1), 1)
    mat2 = np.tile([0, 1, 4, 3, 2, 5, 6, 7, 8], (9, 1))
    assertEqual(neighborAccuracy(mat1, mat2), 0.5)

    print('--> Great Success')
    exit(0)



#unit_test()
