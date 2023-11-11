"""Tools for computing second moments and pi under Wright-Fisher models"""
import scipy.sparse
import scipy.sparse.linalg
import numpy as np
from itertools import chain


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


def build_1d_spatial(theta, migration_rate, num_demes, pop_sizes=1.):
    """
    Get the coefficients of the ODE system for a 1D spatial model

    Considers a model where all of the demes are in a single chain, and
    migration only occurs between adjacent demes at a constant rate.

    Args:
        theta: population scaled mutation rate
        migration_rate: population scaled migration rate between adjacent demes
        num_demes: number of demes in the chain
        pop_sizes: scaled population sizes for each deme. If scalar, then all
            demes have the same size, otherwise it should be a numpy array of
            shape (num_demes,). Defaults to all demes having size 1.

    Returns:
        (M, v) where the dynamics of the second moments, x, are described by
        the ODE dx/dt = M @ x + v.  That is, M is the matrix of coefficients
        of the terms in the ODE that multiply the current second moments and
        v is the additive, constant terms in the ODE.
    """

    assert np.all(pop_sizes > 0)

    deme_pop_sizes = np.zeros(num_demes)
    deme_pop_sizes[:] = pop_sizes

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
                moment_mat[this_idx, this_idx] -= 1. / deme_pop_sizes[i]
                const_vec[this_idx] += 0.5 / deme_pop_sizes[i]
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


def build_2d_index(xlen, ylen, extinct_demes=None):
    """
    Creates a map to and from deme indices for a 2D spatial model

    In a 2D spatial model, it makes sense to describe each deme by its latitude
    and longitude. We need to map this to arbitrary deme models where we just
    number demes from 1 to number of demes. This function creates maps to and
    from latitude and longitude to "deme index".

    Args:
        xlen: how many demes across is the space
        ylen: how many demes from top to bottom is the space
        extinct_demes: numpy array of shape (xlen, ylen) where entry i, j is
            True if the deme at spatial position i, j is totally extinct,
            otherwise False. Defaults to None, in which case there are not
            extinct demes.


    Returns:
        (lat_lon_map, inverse_lat_lon_map), where lat_lon_map takes a
        "deme index" and then returns the latitude and longitude corresponding
        to that deme, and inverse_lat_lon_map takes a tuple of latitude and
        longitude and returns the deme index for the deme at that spatial
        position.
    """
    if extinct_demes is None:
        extinct_demes = np.zeros((xlen, ylen), dtype=bool)
    assert extinct_demes.shape == (xlen, ylen)

    idx_to_xy = []
    xy_to_idx = {}
    for i in range(xlen):
        for j in range(ylen):
            if extinct_demes[i, j]:
                continue
            xy_to_idx[(i, j)] = len(idx_to_xy)
            idx_to_xy.append((i, j))
    return idx_to_xy, xy_to_idx


def build_2d_spatial(
    theta,
    migration_rate,
    xlen,
    ylen,
    pop_sizes=1.,
    extinct_demes=None
):
    """
    Get the coefficients of the ODE system for a 2D spatial model

    Considers a model where demes are arrayed in a 2D grid, and migration only
    occurs between adjacen demes at a constant rate.

    Args:
        theta: population scaled mutation rate
        migration_rate: population scaled migration rate between adjacent demes
        xlen: how wide the habitat is in terms of number of demes
        ylen: how long the habitat is in terms of number of demes
        pop_sizes: scaled population sizes for each deme. If scalar, then all
            demes have the same size, otherwise it should be a numpy array of
            shape (xlen, ylen). Defaults to all demes having size 1.
        extinct_demes: numpy array of shape (xlen, ylen) where entry i, j is
            True if the deme at spatial position i, j is totally extinct,
            otherwise False. Defaults to None, in which case there are not
            extinct demes.

    Returns:
        (M, v) where the dynamics of the secondmoments, x, are described by the
        ODE dx/dt = M @ x + v. That is, M is hte matrix of coefficients of the
        terms in the ODE that multiply the current second moments and v is the
        additive, constant terms in the ODE.
    """

    assert np.all(pop_sizes > 0)

    if extinct_demes is None:
        extinct_demes = np.zeros((xlen, ylen), dtype=bool)
    assert extinct_demes.shape == (xlen, ylen)

    deme_pop_sizes = np.zeros((xlen, ylen))
    deme_pop_sizes[:, :] = pop_sizes

    idx_to_xy, xy_to_idx = build_2d_index(xlen, ylen, extinct_demes)
    idx_to_deme, deme_to_idx = build_deme_index(len(idx_to_xy))
    sq_num_demes = len(idx_to_deme)
    moment_mat = scipy.sparse.dok_matrix((sq_num_demes, sq_num_demes),
                                         dtype=np.float64)
    const_vec = np.ones(sq_num_demes, dtype=np.float64) * theta/2
    for x1 in range(xlen):
        for y1 in range(ylen):
            if extinct_demes[x1, y1]:
                continue
            for x2 in range(xlen):
                for y2 in range(ylen):
                    if extinct_demes[x2, y2]:
                        continue

                    idx1 = xy_to_idx[(x1, y1)]
                    idx2 = xy_to_idx[(x2, y2)]
                    if idx2 < idx1:
                        continue
                    this_idx = deme_to_idx[(idx1, idx2)]
                    moment_mat[this_idx, this_idx] = -2*theta
                    if idx1 == idx2:
                        moment_mat[this_idx, this_idx] -= (
                            1. / deme_pop_sizes[x1, y1]
                        )
                        const_vec[this_idx] += 0.5 / deme_pop_sizes[x1, y1]
                    # migration
                    if x1 > 0 and not extinct_demes[x1-1, y1]:
                        my_idx = deme_to_idx[(xy_to_idx[(x1-1, y1)],
                                              idx2)]
                        moment_mat[this_idx, my_idx] += migration_rate
                        moment_mat[this_idx, this_idx] -= migration_rate
                    if y1 > 0 and not extinct_demes[x1, y1-1]:
                        my_idx = deme_to_idx[(xy_to_idx[(x1, y1-1)],
                                              idx2)]
                        moment_mat[this_idx, my_idx] += migration_rate
                        moment_mat[this_idx, this_idx] -= migration_rate
                    if x1 < xlen - 1 and not extinct_demes[x1+1, y1]:
                        my_idx = deme_to_idx[(xy_to_idx[(x1+1, y1)],
                                              idx2)]
                        moment_mat[this_idx, my_idx] += migration_rate
                        moment_mat[this_idx, this_idx] -= migration_rate
                    if y1 < ylen - 1 and not extinct_demes[x1, y1+1]:
                        my_idx = deme_to_idx[(xy_to_idx[(x1, y1+1)],
                                              idx2)]
                        moment_mat[this_idx, my_idx] += migration_rate
                        moment_mat[this_idx, this_idx] -= migration_rate
                    if x2 > 0 and not extinct_demes[x2-1, y2]:
                        my_idx = deme_to_idx[(xy_to_idx[(x2-1, y2)],
                                              idx1)]
                        moment_mat[this_idx, my_idx] += migration_rate
                        moment_mat[this_idx, this_idx] -= migration_rate
                    if y2 > 0 and not extinct_demes[x2, y2-1]:
                        my_idx = deme_to_idx[(xy_to_idx[(x2, y2-1)],
                                              idx1)]
                        moment_mat[this_idx, my_idx] += migration_rate
                        moment_mat[this_idx, this_idx] -= migration_rate
                    if x2 < xlen - 1 and not extinct_demes[x2+1, y2]:
                        my_idx = deme_to_idx[(xy_to_idx[(x2+1, y2)],
                                              idx1)]
                        moment_mat[this_idx, my_idx] += migration_rate
                        moment_mat[this_idx, this_idx] -= migration_rate
                    if y2 < ylen - 1 and not extinct_demes[x2, y2+1]:
                        my_idx = deme_to_idx[(xy_to_idx[(x2, y2+1)],
                                              idx1)]
                        moment_mat[this_idx, my_idx] += migration_rate
                        moment_mat[this_idx, this_idx] -= migration_rate

    return moment_mat.tocsr(), const_vec


def build_1d_laplace(num_demes, m):
    """
    Construct a migration matrix for a 1D chain with homogenous migration

    Assumes that demes are spatially ordered 1, ..., num_demes, migration
    only occurs between adjacent demes with constant rate m and then
    constructs the corresponding num_demes x num_demes migration matrix.

    Args:
        num_demes: the number of demes
        m: the population scaled migration raet

    Returns:
        A num_demes by num_demes numpy array where entry i, j is the migration
        rate between demes i and j.
    """
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
    """
    Construct a migration matrix for a 2D spatial model

    Assumes that demes are ordered according to build_2d_index and that
    migration only occurs between adjacent demes (demes that differ by
    latitude or longitude (but not both) by 1).

    Args:
        xlen: how many demes across is the space
        ylen: how many demes from tp to bottom is the space

    Returns:
        A (xlen * ylen) by (xlen * ylen) numpy array where entry i, j is the
        migration rate between deme i and deme j, where the demes are ordered
        according to build_2d_index.
    """
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


def build_arbitrary(theta, m_mat, pop_sizes=1.):
    """
    Get the coefficients of the ODE system for an arbitrary deme model

    Args:
        theta: population scaled mutation rate
        m_mat: a numpy array where entry i, j is the population scaled
            migration rate from demes i to j. Must be a rate matrix:
            all off-diagonals must be non-negative and rows must sum to zero.
        pop_sizes: scaled population sizes for each deme. If scalar, then all
            demes have the same size, otherwise it should be a numpy array of
            shape (num_demes,). Defaults to all demes having size 1.


    Returns:
        (M, v) where the dynamics of the second moments, x, are described by
        the ODE dx/dt = M @ x + v. That is, M is the matrix of coefficients of
        the terms in the ODE that multiply the current second moments and v is
        the additive, constant terms in the ODE.

    """
    assert np.all(pop_sizes > 0.)
    assert len(m_mat.shape) == 2
    assert m_mat.shape[0] == m_mat.shape[1]
    assert np.allclose(m_mat.sum(axis=1), 0)

    deme_pop_sizes = np.zeros(m_mat.shape[0])
    deme_pop_sizes[:] = pop_sizes

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
                moment_mat[this_idx, this_idx] -= 1 / deme_pop_sizes[i]
                const_vec[this_idx] += 0.5 / deme_pop_sizes[i]
            for k in range(num_demes):
                ki_idx = deme_to_idx[(k, i)]
                kj_idx = deme_to_idx[(k, j)]
                moment_mat[this_idx, ki_idx] += m_mat[k, j]
                moment_mat[this_idx, kj_idx] += m_mat[k, i]

    return moment_mat.tocsr(), const_vec


def compute_equilibrium(moment_mat, const_vec):
    """
    Get the equilibrium of a linear ODE

    Solves the steady state of a linear ODE on a variable x described by
    dx/dt = moment_mat @ x + const_vec.

    Args:
        moment_mat: the matrix of coefficients of the terms in the ODE that
            multiply x.
        const_vec: the vector of constant additive terms in the ODE.

    Returns:
        A 1d numpy array containing the equilibrium solution of the ODE

    """
    return scipy.sparse.linalg.spsolve(moment_mat, -const_vec)


def evolve_forward(moment_mat, const_vec, curr_moments, time):
    """
    Solve an ODE forward for a set amount of time

    Integrates a linear ODE described by
    d(curr_moments)/dt = moment_mat @ curr_moments + const_vec
    forward in time.

    Args:
        moment_mat: the matrix of coefficients of the terms in the ODE that
            multiply curr_moments.
        const_vec: the vector of constant additive terms in the ODE.
        curr_moments: the current second moments of the model
        time: the amount of time to integrate the ODE forward

    Returns:
        A 1d numpy array containing the second moments after evolving forward
        for time amount of time.
    """
    m_inv_v = -compute_equilibrium(moment_mat, const_vec)
    evolved = scipy.sparse.linalg.expm_multiply(
        moment_mat * time, curr_moments + m_inv_v
    )
    return evolved - m_inv_v


def num_demes_from_num_moments(k):
    """Get the number of demes from the number of second moments"""
    return int(round(0.5 * (np.sqrt(8*k + 1) - 1)))


def compute_pi(curr_moments, demes):
    """
    Compute the average heterozygosity from the second moments of demes

    Args:
        curr_moments: a 1d numpy array of the second moments of the allele
            frequencies across all demes
        demes: a list of the demes from which individuals should be sampled.

    Returns:
        The average pairwise heterozygosity when individuals are uniformly
        sampled from the specified demes.
    """
    num_demes = num_demes_from_num_moments(len(curr_moments))
    idx_to_deme, deme_to_idx = build_deme_index(num_demes)

    total_pi = 0.
    for i in demes:
        for j in demes:
            this_moment = curr_moments[deme_to_idx[(i, j)]]
            total_pi += 2 / (len(demes)**2) * (0.5 - this_moment)

    return total_pi


def compute_fst_hudson(curr_moments, demes_1, demes_2):
    """
    Compute Hudson's Fst between demes_1 and demes_2

    Args:
        curr_moments: a 1d numpy array of the second moments of the allele
            frequencies across all demes
        demes_1: a list of the demes from which the first set of individuals
            should be drawn.
        demes_2: a list of the demes from which the second set of individuals
            should be drawn.

    Returns:
        Hudson's Fst between the set of individuals in demes_1 and demes_2.
        This estimator does not weight by population size, so pi_within is just
        the average of pi in deme_1 and pi in deme_2.
    """

    # make sure these are separate demes
    assert len(set(demes_1).intersection(set(demes_2))) == 0

    curr_moments = np.copy(curr_moments)

    pi_within_1 = compute_pi(curr_moments, demes_1)
    pi_within_2 = compute_pi(curr_moments, demes_2)
    pi_within = 0.5 * (pi_within_1 + pi_within_2)

    num_demes = num_demes_from_num_moments(len(curr_moments))
    idx_to_deme, deme_to_idx = build_deme_index(num_demes)
    pi_between = 0.
    for i in demes_1:
        for j in demes_2:
            # The next line of code is based on the following algebraic
            # simplification
            # E[p_1 * (1-p_2) + (1-p_1) * p_2]
            # = E[p_1 + p_2 - 2 * p_1 * p_2]
            # = 0.5 + 0.5 - 2 * second_moment
            pi_between += 1 - 2 * curr_moments[deme_to_idx[(i, j)]]

    pi_between = pi_between / len(demes_1) / len(demes_2)
    return 1. - pi_within / pi_between


def compute_fst_nei(curr_moments, demes_1, demes_2):
    """
    Compute Nei's Fst between demes_1 and demes_2

    Args:
        curr_moments: a 1d numpy array of the second moments of the allele
            frequencies across all demes
        demes_1: a list of the demes from which the first set of individuals
            should be drawn.
        demes_2: a list of the demes from which the second set of individuals
            should be drawn.

    Returns:
        Nei's Fst between the set of individuals in demes_1 and demes_2.
        Assumes that the number of individuals is the same across all demes.
        That is, pi_within and pi_total are weighted by the number of demes
        within each set of demes.
    """

    # make sure these are separate demes
    assert len(set(demes_1).intersection(set(demes_2))) == 0

    all_demes = list(demes_1) + list(demes_2)

    curr_moments = np.copy(curr_moments)

    pi_within_1 = compute_pi(curr_moments, demes_1)
    pi_within_2 = compute_pi(curr_moments, demes_2)

    pi_within = (
        pi_within_1 * len(demes_1) / len(all_demes)
        + pi_within_2 * len(demes_2) / len(all_demes)
    )

    pi_total = compute_pi(curr_moments, all_demes)

    return 1. - pi_within / pi_total


def reseed(indices_to_reseed, source_indices, curr_moments):
    """
    Replace the individuals in a set of demes with individuals from other demes

    Takes a set of current second moments and computes what the second moments
    would be if we were to replace all of the individuals in each set of target
    demes with a random sample from the corresponding set of source demes.

    Args:
        indices_to_reseed: a list of lists of the demes whose individuals
            will be replaced.
        source_indices: a list of lists of the demes from which to sample the
            individuals used to reseed the demes specified by
            indices_to_reseed.
        curr_moments: a 1d numpy array of the second moments of the allele
            frequencies across all demes.

    Returns:
        a 1d numpy array containing the second moments of the allele
        frequencies across all demes after the individuals in
        indices_to_reseed have been replace by individuals chosen at random
        from source_indices.
    """
    assert len(indices_to_reseed) == len(source_indices)
    flat_target = list(chain.from_iterable(indices_to_reseed))
    flat_source = list(chain.from_iterable(source_indices))
    assert len(flat_target) == len(set(flat_target))
    assert len(set(flat_target) & set(flat_source)) == 0

    curr_moments = np.copy(curr_moments)
    num_demes = num_demes_from_num_moments(len(curr_moments))
    idx_to_deme, deme_to_idx = build_deme_index(num_demes)

    inner_moments = np.zeros(
        (len(source_indices), len(source_indices))
    )
    for idx1, source_index_set_1 in enumerate(source_indices):
        for idx2, source_index_set_2 in enumerate(source_indices):
            avg_moments = 0.
            for i in source_index_set_1:
                for j in source_index_set_2:
                    avg_moments += curr_moments[deme_to_idx[(i, j)]]
            avg_moments /= len(source_index_set_1)*len(source_index_set_2)
            inner_moments[idx1, idx2] = avg_moments

    target_map = {}
    for idx, target_set in enumerate(indices_to_reseed):
        for deme in target_set:
            target_map[deme] = idx

    for target_set in indices_to_reseed:
        for i in target_set:
            for k in range(num_demes):
                curr_moments[deme_to_idx[(i, k)]] = 0.
                if k in target_map:
                    curr_moments[deme_to_idx[(i, k)]] = inner_moments[
                        target_map[i], target_map[k]
                    ]
                else:
                    for j in source_indices[target_map[i]]:
                        curr_moments[deme_to_idx[(i, k)]] += (
                            curr_moments[deme_to_idx[(j, k)]]
                            / len(source_indices[target_map[i]])
                        )
    return curr_moments


def get_moments(curr_moments, demes):
    """
    Get the second moments for a (sub)set of demes

    From an array of second moments, we may want to pull out only those second
    moments that correspond to pairs of demes both within some subset.

    Args:
        curr_moments: a 1d numpy array of the second moments of the allele
            frequencies across all demes.
        demes: a list of the demes for which we want second moments that only
            invole these demes. If an entry is None, then this is a new deme
            not present in the current set of demes and all moments involving
            this entry will be np.nan.

    Returns:
        a 1d numpy array containing the second moments of the allele
        frequencies across only pairs of demes from the provided list demes.
    """
    new_idx_to_deme, new_deme_to_idx = build_deme_index(len(demes))
    old_idx_to_deme, old_deme_to_idx = build_deme_index(
        num_demes_from_num_moments(len(curr_moments))
    )
    to_return = np.zeros(len(new_idx_to_deme))
    for new_idx in range(len(to_return)):
        new_i, new_j = new_idx_to_deme[new_idx]
        i = demes[new_i]
        j = demes[new_j]
        if i is None or j is None:
            to_return[new_idx] = np.nan
        else:
            old_idx = old_deme_to_idx[(i, j)]
            to_return[new_idx] = curr_moments[old_idx]
    return to_return


def get_moments_2d(curr_moments, old_extinct_demes, new_extinct_demes):
    """
    Get the second moments for a subset of demes in a 2D landscape

    Similar to get_moments, but specific to the 2D case where instead of
    directly indexing demes, we may just want to use 2D maps of which demes are
    extinct.

    Args:
        curr_moments: a 1d numpy array of the second moments of the allele
            frequencies across all demes.
        old_extinct_demes: numpy array of shape (xlen, ylen) where entry i, j
            is True if the deme at spatial position i, j is currently extinct
            otherwise False.
        new_extinct_demes: numpy array of shape (xlen, ylen) where entry i, j
            is True if the deme at spatial position i, j will be extinct
            after this event, othrwie False.

    Returns:
        a 1d numpy array containing the second moments of the allele
        frequencies across only pairs of demes that are not extinct in
        new_extinct_demes.
    """
    assert (
        old_extinct_demes is None
        or new_extinct_demes is None
        or old_extinct_demes.shape == new_extinct_demes.shape
    )

    if old_extinct_demes is None and new_extinct_demes is None:
        return curr_moments

    if old_extinct_demes is None:
        x_len, y_len = new_extinct_demes.shape
    else:
        x_len, y_len = old_extinct_demes.shape

    old_idx_to_xy, old_xy_to_idx = build_2d_index(
        x_len, y_len, old_extinct_demes
    )

    new_idx_to_xy, new_xy_to_idx = build_2d_index(
        x_len, y_len, new_extinct_demes
    )

    demes = []
    for x, y in new_idx_to_xy:
        if old_extinct_demes is None or not old_extinct_demes[x, y]:
            demes.append(old_xy_to_idx[(x, y)])
        else:
            demes.append(None)

    return get_moments(curr_moments, demes)
