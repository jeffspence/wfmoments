"""Tools for computing second moments and pi under Wright-Fisher models"""
import scipy.sparse
import scipy.sparse.linalg
import numpy as np


def build_deme_index(num_demes):
    """
    Creates a map to and from indices for vecorized second moments.

    In a model with multiple demes, there are distinct second moments
    for each unordered combination of two not necessarily unique demes.
    The dynamics of these second moments are described by a linear ODE,
    however, so it is convenient to represent all of these distinct second
    moments as a vector and implement the ODE with matrices. As such, we need
    a mapping to and from the elements of this vector to the corresponding
    pair of demes.

    Args:
        num_demes: the total number of demes

    Returns:
        (deme_map, inverse_deme_map), where deme_map takes an index in the
        vectorized second moment and returns a tuple with two demes whose
        second moment is represented by that entry of the vector. The demes are
        ordered so that the second listed deme never has a lower index than the
        first. inverse_deme_map is a dict that maps tuples of demes
        to vector indices, such that inverse_deme_map[(i, j)] is the vector
        index corresponding to the second moment of the frequencies
        in demes i and j. For inverse_deme_map, order of the demes does not
        matter.
    """
    to_return = []
    to_return_inv = {}
    for i in range(num_demes):
        for j in range(i, num_demes):
            to_return_inv[(i, j)] = len(to_return)
            to_return_inv[(j, i)] = len(to_return)
            to_return.append((i, j))
    return to_return, to_return_inv


def build_1d_spatial(theta, migration_rate, num_demes):
    idx_to_deme, deme_to_idx = build_deme_index(num_demes)
    sq_num_demes = len(idx_to_deme)
    moment_mat = scipy.sparse.dok_matrix((sq_num_demes, sq_num_demes),
                                         dtype=np.float64)
    const_vec = np.ones(sq_num_demes, dtype=np.float64) * theta/2
    for i in range(num_demes):
        for j in range(i, num_demes):
            this_idx = deme_to_idx[(i, j)]
            moment_mat[this_idx, this_idx] = -2*theta
            if i == j:
                moment_mat[this_idx, this_idx] -= 1.
                const_vec[this_idx] += 0.5
            # migration
            if i > 0:
                im1_idx = deme_to_idx[(i-1, j)]
                moment_mat[this_idx, im1_idx] += migration_rate
                moment_mat[this_idx, this_idx] -= migration_rate
            if i < num_demes - 1:
                ip1_idx = deme_to_idx[(i+1, j)]
                moment_mat[this_idx, ip1_idx] += migration_rate
                moment_mat[this_idx, this_idx] -= migration_rate
            if j > 0:
                jm1_idx = deme_to_idx[(i, j-1)]
                moment_mat[this_idx, jm1_idx] += migration_rate
                moment_mat[this_idx, this_idx] -= migration_rate
            if j < num_demes - 1:
                jp1_idx = deme_to_idx[(i, j+1)]
                moment_mat[this_idx, jp1_idx] += migration_rate
                moment_mat[this_idx, this_idx] -= migration_rate

    return moment_mat.tocsr(), const_vec


def build_2d_index(xlen, ylen):
    idx_to_xy = []
    xy_to_idx = {}
    for i in range(xlen):
        for j in range(ylen):
            xy_to_idx[(i, j)] = len(idx_to_xy)
            idx_to_xy.append((i, j))
    return idx_to_xy, xy_to_idx


def build_2d_spatial(theta, migration_rate, xlen, ylen):
    idx_to_xy, xy_to_idx = build_2d_index(xlen, ylen)
    idx_to_deme, deme_to_idx = build_deme_index(xlen*ylen)
    sq_num_demes = len(idx_to_deme)
    moment_mat = scipy.sparse.dok_matrix((sq_num_demes, sq_num_demes),
                                         dtype=np.float64)
    const_vec = np.ones(sq_num_demes, dtype=np.float64) * theta/2
    for x1 in range(xlen):
        for y1 in range(ylen):
            for x2 in range(xlen):
                for y2 in range(ylen):
                    idx1 = xy_to_idx[(x1, y1)]
                    idx2 = xy_to_idx[(x2, y2)]
                    if idx2 < idx1:
                        continue
                    this_idx = deme_to_idx[(idx1, idx2)]
                    moment_mat[this_idx, this_idx] = -2*theta
                    if idx1 == idx2:
                        moment_mat[this_idx, this_idx] -= 1.0
                        const_vec[this_idx] += 0.5
                    # migration
                    if x1 > 0:
                        my_idx = deme_to_idx[(xy_to_idx[(x1-1, y1)],
                                              idx2)]
                        moment_mat[this_idx, my_idx] += migration_rate
                        moment_mat[this_idx, this_idx] -= migration_rate
                    if y1 > 0:
                        my_idx = deme_to_idx[(xy_to_idx[(x1, y1-1)],
                                              idx2)]
                        moment_mat[this_idx, my_idx] += migration_rate
                        moment_mat[this_idx, this_idx] -= migration_rate
                    if x1 < xlen - 1:
                        my_idx = deme_to_idx[(xy_to_idx[(x1+1, y1)],
                                              idx2)]
                        moment_mat[this_idx, my_idx] += migration_rate
                        moment_mat[this_idx, this_idx] -= migration_rate
                    if y1 < ylen - 1:
                        my_idx = deme_to_idx[(xy_to_idx[(x1, y1+1)],
                                              idx2)]
                        moment_mat[this_idx, my_idx] += migration_rate
                        moment_mat[this_idx, this_idx] -= migration_rate
                    if x2 > 0:
                        my_idx = deme_to_idx[(xy_to_idx[(x2-1, y2)],
                                              idx1)]
                        moment_mat[this_idx, my_idx] += migration_rate
                        moment_mat[this_idx, this_idx] -= migration_rate
                    if y2 > 0:
                        my_idx = deme_to_idx[(xy_to_idx[(x2, y2-1)],
                                              idx1)]
                        moment_mat[this_idx, my_idx] += migration_rate
                        moment_mat[this_idx, this_idx] -= migration_rate
                    if x2 < xlen - 1:
                        my_idx = deme_to_idx[(xy_to_idx[(x2+1, y2)],
                                              idx1)]
                        moment_mat[this_idx, my_idx] += migration_rate
                        moment_mat[this_idx, this_idx] -= migration_rate
                    if y2 < ylen - 1:
                        my_idx = deme_to_idx[(xy_to_idx[(x2, y2+1)],
                                              idx1)]
                        moment_mat[this_idx, my_idx] += migration_rate
                        moment_mat[this_idx, this_idx] -= migration_rate

    return moment_mat.tocsr(), const_vec


def build_1d_laplace(num_demes, m):
    to_return = np.zeros((num_demes, num_demes))
    for i in range(num_demes):
        if i > 0:
            to_return[i, i-1] = m
            to_return[i, i] -= m
        if i < num_demes - 1:
            to_return[i, i+1] = m
            to_return[i, i] -= m
    return to_return


def build_2d_laplace(xlen, ylen, m):
    num_demes = xlen * ylen
    to_return = np.zeros((num_demes, num_demes))
    idx_to_xy, xy_to_idx = build_2d_index(xlen, ylen)

    for i in range(xlen):
        for j in range(ylen):
            this_idx = xy_to_idx[(i, j)]
            if i > 0:
                to_idx = xy_to_idx[(i-1, j)]
                to_return[this_idx, to_idx] = m
                to_return[this_idx, this_idx] -= m
            if j > 0:
                to_idx = xy_to_idx[(i, j-1)]
                to_return[this_idx, to_idx] = m
                to_return[this_idx, this_idx] -= m
            if i < xlen - 1:
                to_idx = xy_to_idx[(i+1, j)]
                to_return[this_idx, to_idx] = m
                to_return[this_idx, this_idx] -= m
            if j < ylen - 1:
                to_idx = xy_to_idx[(i, j+1)]
                to_return[this_idx, to_idx] = m
                to_return[this_idx, this_idx] -= m
    return to_return


def build_arbitrary(theta, m_mat):
    assert len(m_mat.shape) == 2
    assert m_mat.shape[0] == m_mat.shape[1]
    assert np.allclose(m_mat.sum(axis=1), 0)
    num_demes = m_mat.shape[0]
    idx_to_deme, deme_to_idx = build_deme_index(num_demes)
    sq_num_demes = len(idx_to_deme)
    moment_mat = scipy.sparse.dok_matrix((sq_num_demes, sq_num_demes),
                                         dtype=np.float64)
    const_vec = np.ones(sq_num_demes, dtype=np.float64) * theta/2
    for i in range(num_demes):
        for j in range(i, num_demes):
            this_idx = deme_to_idx[(i, j)]
            moment_mat[this_idx, this_idx] = -2*theta
            if i == j:
                moment_mat[this_idx, this_idx] -= 1.0
                const_vec[this_idx] += 0.5
            for k in range(num_demes):
                ki_idx = deme_to_idx[(k, i)]
                kj_idx = deme_to_idx[(k, j)]
                moment_mat[this_idx, ki_idx] += m_mat[k, j]
                moment_mat[this_idx, kj_idx] += m_mat[k, i]

    return moment_mat.tocsr(), const_vec


def compute_equilibrium(moment_mat, const_vec):
    return scipy.sparse.linalg.spsolve(moment_mat, -const_vec)


def evolve_forward(moment_mat, const_vec, curr_moments, time):
    m_inv_v = -compute_equilibrium(moment_mat, const_vec)
    evolved = scipy.sparse.linalg.expm_multiply(
        moment_mat * time, curr_moments + m_inv_v
    )
    return evolved - m_inv_v


def num_demes_from_moments(k):
    return int(round(0.5 * (np.sqrt(8*k + 1) - 1)))


def compute_pi(curr_moments, demes):
    num_demes = num_demes_from_moments(len(curr_moments))
    idx_to_deme, deme_to_idx = build_deme_index(num_demes)

    total_pi = 0.
    for i in demes:
        for j in demes:
            this_moment = curr_moments[deme_to_idx[(i, j)]]
            total_pi += 2 / (len(demes)**2) * (0.5 - this_moment)

    return total_pi


def reseed(indices_to_reseed, source_indices, curr_moments):
    curr_moments = np.copy(curr_moments)
    num_demes = num_demes_from_moments(len(curr_moments))
    idx_to_deme, deme_to_idx = build_deme_index(num_demes)

    inner_moments = 0.
    for i in source_indices:
        for j in source_indices:
            inner_moments += curr_moments[deme_to_idx[(i, j)]]
    inner_moments /= len(source_indices)**2

    for i in indices_to_reseed:
        for k in range(num_demes):
            curr_moments[deme_to_idx[(i, k)]] = 0.
            if k in indices_to_reseed:
                curr_moments[deme_to_idx[(i, k)]] = inner_moments
            else:
                for j in source_indices:
                    curr_moments[deme_to_idx[(i, k)]] += (
                        curr_moments[deme_to_idx[(j, k)]]
                        / len(source_indices)
                    )
    return curr_moments


def get_moments(curr_moments, demes):
    new_idx_to_deme, new_deme_to_idx = build_deme_index(len(demes))
    old_idx_to_deme, old_deme_to_idx = build_deme_index(
        num_demes_from_moments(len(curr_moments))
    )
    to_return = np.zeros(len(new_idx_to_deme))
    for new_idx in range(len(to_return)):
        new_i, new_j = new_idx_to_deme[new_idx]
        i = demes[new_i]
        j = demes[new_j]
        old_idx = old_deme_to_idx[(i, j)]
        to_return[new_idx] = curr_moments[old_idx]
    return to_return
