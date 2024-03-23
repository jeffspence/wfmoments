import wfmoments
import numpy as np


def test_build_deme_index():
    idx_to_deme, deme_to_idx = wfmoments.build_deme_index(2)
    assert idx_to_deme[0] == (0, 0)
    assert idx_to_deme[1] == (0, 1)
    assert idx_to_deme[2] == (1, 1)
    assert len(idx_to_deme) == 3
    assert deme_to_idx[(0, 0)] == 0
    assert deme_to_idx[(0, 1)] == 1
    assert deme_to_idx[(1, 0)] == 1
    assert deme_to_idx[(1, 1)] == 2

    idx_to_deme, deme_to_idx = wfmoments.build_deme_index(100)
    for idx, (i, j) in enumerate(idx_to_deme):
        assert deme_to_idx[(i, j)] == idx
        assert deme_to_idx[(j, i)] == idx


# tests build_1d_spatial, build_1d_laplace, and build_arbitrary
def test_build_1d_spatial():
    m, v = wfmoments.build_1d_spatial(1e-1, 1e-2, 10)
    m_mat = wfmoments.build_1d_laplace(10, 1e-2)
    m_check, v_check = wfmoments.build_arbitrary(1e-1, m_mat)
    x = np.random.random(m.shape[1])
    assert np.allclose(m.dot(x), m_check.dot(x))
    assert np.allclose(v, v_check)

    m_pop, v_pop = wfmoments.build_1d_spatial(1e-1, 1e-2, 10, 1.)
    assert np.allclose(m_pop.dot(x), m.dot(x))
    assert np.allclose(v_pop, v)

    m_array, v_array = wfmoments.build_1d_spatial(
        1e-1, 1e-2, 10, np.ones(10)
    )
    assert np.allclose(m_pop.dot(x), m_array.dot(x))
    assert np.allclose(v_pop, v_array)

    m_big, v_big = wfmoments.build_1d_spatial(1e-2, 1e-3, 10, 10.)
    orig_eq = wfmoments.compute_equilibrium(m, v)
    new_eq = wfmoments.compute_equilibrium(m_big, v_big)
    assert np.allclose(orig_eq, new_eq)

    pop_sizes = np.random.random(10) + 1
    m_array, v_array = wfmoments.build_1d_spatial(
        1e-1, 1e-2, 10, pop_sizes
    )
    m_arb, v_arb = wfmoments.build_arbitrary(1e-1, m_mat, pop_sizes)
    assert np.allclose(m_array.dot(x), m_arb.dot(x))
    assert np.allclose(v_array, v_arb)


def test_build_2d_index():
    idx_to_xy, xy_to_idx = wfmoments.build_2d_index(5, 7)
    assert len(idx_to_xy) == 35
    for x in range(5):
        for y in range(7):
            assert (x, y) in idx_to_xy
            assert (x, y) in xy_to_idx
    for i in range(35):
        assert xy_to_idx[idx_to_xy[i]] == i


# tests build_2d_spatial, build_2d_laplace, and build_arbitrary
def test_build_2d_spatial():
    m, v = wfmoments.build_2d_spatial(1e-1, 1e-2, 5, 7)
    m_mat = wfmoments.build_2d_laplace(5, 7, 1e-2)
    m_check, v_check = wfmoments.build_arbitrary(1e-1, m_mat)
    x = np.random.random(m.shape[1])
    assert np.allclose(m.dot(x), m_check.dot(x))
    assert np.allclose(v, v_check)

    m_pop, v_pop = wfmoments.build_2d_spatial(1e-1, 1e-2, 5, 7, 1.)
    assert np.allclose(m_pop.dot(x), m.dot(x))
    assert np.allclose(v_pop, v)

    m_big, v_big = wfmoments.build_2d_spatial(1e-2, 1e-3, 5, 7, 10.)
    orig_eq = wfmoments.compute_equilibrium(m, v)
    new_eq = wfmoments.compute_equilibrium(m_big, v_big)
    assert np.allclose(orig_eq, new_eq)

    pop_sizes = np.random.random((5, 7)) + 1
    m, v = wfmoments.build_2d_spatial(1e-1, 1e-2, 5, 7, pop_sizes)
    check_idx_map, _ = wfmoments.build_2d_index(5, 7)
    flat_pop_sizes = np.zeros(35)
    for i in range(35):
        flat_pop_sizes[i] = pop_sizes[check_idx_map[i]]
    m_check, v_check = wfmoments.build_arbitrary(1e-1, m_mat, flat_pop_sizes)
    assert np.allclose(m.dot(x), m_check.dot(x))
    assert np.allclose(v, v_check)

    # test extincting some demes
    extinct_demes = np.ones((5, 7), dtype=bool)
    extinct_demes[:, 0] = False

    m_ex, v_ex = wfmoments.build_2d_spatial(
        1e-1, 1e-2, 5, 7, extinct_demes=extinct_demes
    )
    m_1d, v_1d = wfmoments.build_1d_spatial(1e-1, 1e-2, 5)
    x_short = np.random.random(m_ex.shape[1])
    assert np.allclose(m_ex.dot(x_short), m_1d.dot(x_short))
    assert np.allclose(v_ex, v_1d)

    extinct_demes = np.random.choice([False, True], size=(5, 7))
    extinct_demes[0, 0] = False  # make sure at least one deme is still there
    m_ex, v_ex = wfmoments.build_2d_spatial(
        1e-1, 1e-2, 5, 7, extinct_demes=extinct_demes
    )
    x_short = np.random.random(m_ex.shape[1])

    kept_demes = np.zeros(m_mat.shape[0], dtype=bool)
    idx_to_xy, _ = wfmoments.build_2d_index(5, 7)
    for idx in range(m_mat.shape[0]):
        i, j = idx_to_xy[idx]
        if extinct_demes[i, j]:
            continue
        kept_demes[idx] = True
    m_mat_restricted = m_mat[np.ix_(kept_demes, kept_demes)]
    m_mat_restricted[
        np.arange(m_mat_restricted.shape[0]),
        np.arange(m_mat_restricted.shape[0])
    ] = 0
    m_mat_restricted[
        np.arange(m_mat_restricted.shape[0]),
        np.arange(m_mat_restricted.shape[0])
    ] = - m_mat_restricted.sum(axis=1)
    m_check, v_check = wfmoments.build_arbitrary(1e-1, m_mat_restricted)
    assert np.allclose(m_ex.dot(x_short), m_check.dot(x_short))
    assert np.allclose(v_ex, v_check)


# also tests compute_equilibrium
def test_evolve_forward():
    m, v = wfmoments.build_1d_spatial(1e-1, 1e-2, 10)
    x = np.zeros(m.shape[1])
    equilibrium = wfmoments.compute_equilibrium(m, v)
    equilibrium_check = wfmoments.evolve_forward(m, v, x, 1000)
    assert np.allclose(equilibrium, equilibrium_check)
    assert np.all(equilibrium <= 0.5)
    assert np.all(equilibrium >= 0)

    short_x = wfmoments.evolve_forward(m, v, x, 1e-100)
    assert np.allclose(x, short_x)

    medium_x = wfmoments.evolve_forward(m, v, x, 0.1)

    m_inv_v = -wfmoments.compute_equilibrium(m, v)
    curr_vec = np.copy(m_inv_v)
    curr_den = 1.
    check_vec = np.zeros_like(m_inv_v)
    for i in range(1, 10):
        curr_vec = m.dot(curr_vec)
        check_vec += curr_vec / curr_den * (0.1**i)
        curr_den *= (i+1)
    assert np.allclose(check_vec, medium_x)


def test_compute_equilibrium():
    m, v = wfmoments.build_1d_spatial(1e-4, 1e-2, 2)
    eq = wfmoments.compute_equilibrium(m, v)
    eq_check = np.linalg.solve(m.todense(), -v)
    assert np.allclose(eq, eq_check)
    assert np.all(eq <= 0.5)
    assert np.all(eq >= 0)
    eq = wfmoments.compute_equilibrium(m, v, direct=True)
    assert np.allclose(eq, eq_check)
    assert np.all(eq <= 0.5)
    assert np.all(eq >= 0)
    for i in range(10):
        x0 = np.random.random(v.shape[0]) * 500
        eq = wfmoments.compute_equilibrium(m, v, direct=False, x0=x0)
        assert np.allclose(eq, eq_check)
        x0 = np.random.normal(scale=500, size=v.shape[0])
        eq = wfmoments.compute_equilibrium(m, v, direct=False, x0=x0)
        assert np.allclose(eq, eq_check)

    for i in range(10):
        theta = np.random.random()
        mig = np.random.random()
        m, v = wfmoments.build_1d_spatial(theta, mig, 10)
        eq = wfmoments.compute_equilibrium(m, v)
        assert np.all(eq < 0.5)
        assert np.all(eq > 0)


def test_num_demes_from_num_moments():
    for i in range(1, 100):
        k = (i * (i+1)) // 2
        assert wfmoments.num_demes_from_num_moments(k) == i
    i = 100000
    k = (i * (i+1)) // 2
    assert wfmoments.num_demes_from_num_moments(k) == i


def test_build_1d_laplace():
    m = wfmoments.build_1d_laplace(10, 0.1)
    assert np.allclose(m.dot(np.ones(10)), 0)
    assert np.allclose(m.T.dot(np.ones(10)), 0)


def test_compute_pi():
    m, v = wfmoments.build_1d_spatial(1e-3, 1, 1)
    eq = wfmoments.compute_equilibrium(m, v)
    pi = wfmoments.compute_pi(eq, [0])
    assert np.isclose(pi, 2*(0.5 - eq[0]))
    assert pi > 0
    pi = wfmoments.compute_pi(eq, [0], weights=[10])

    curr_pi = pi
    curr_theta = 1e-3
    for i in range(10):
        curr_theta *= 2
        m, v = wfmoments.build_1d_spatial(curr_theta, 1, 1)
        eq = wfmoments.compute_equilibrium(m, v)
        pi = wfmoments.compute_pi(eq, [0])
        assert pi > curr_pi
        curr_pi = pi

    m, v = wfmoments.build_1d_spatial(1e-1, 1e-3, 10)
    eq = wfmoments.compute_equilibrium(m, v)
    assert np.all(eq > 0)
    assert np.all(eq < 0.5)
    pi = wfmoments.compute_pi(eq, [0, 1, 9])
    assert pi > 0

    pi_0 = wfmoments.compute_pi(eq, [0])
    e1 = np.zeros(10)
    e1[0] = 1.
    pi_0_check = wfmoments.compute_pi(eq, range(10), e1)
    assert np.isclose(pi_0, pi_0_check)
    e019 = np.zeros(10)
    e019[[0, 1, 9]] = 100
    pi_check = wfmoments.compute_pi(eq, range(10), e019)
    assert np.isclose(pi, pi_check)

    curr_pi = pi
    curr_theta = 1e-1
    for i in range(10):
        curr_theta *= 2
        m, v = wfmoments.build_1d_spatial(curr_theta, 1e-3, 10)
        eq = wfmoments.compute_equilibrium(m, v)
        pi = wfmoments.compute_pi(eq, [0, 1, 9])
        assert pi > curr_pi
        curr_pi = pi


def test_compute_fst_nei():
    m, v = wfmoments.build_1d_spatial(1e-3, 1e-3, 2)
    eq = wfmoments.compute_equilibrium(m, v)
    fst_check = wfmoments.compute_fst_nei(eq, [0], [1])
    assert np.isclose(
        fst_check,
        1 - wfmoments.compute_pi(eq, [0]) / wfmoments.compute_pi(eq, [0, 1])
    )

    fst_check_weight = wfmoments.compute_fst_nei(
        eq, [0], [1], [0.999], [0.001]
    )
    assert np.isclose(
        fst_check_weight,
        1 - wfmoments.compute_pi(eq, [0])
        / wfmoments.compute_pi(eq, [0, 1], [0.999, 0.001])
    )
    mig_mat = np.zeros((4, 4))
    mig_mat[0, 0] = -1000.002
    mig_mat[0, 1] = 1000
    mig_mat[0, 2] = 0.001
    mig_mat[0, 3] = 0.001
    mig_mat[1, 0] = 1000
    mig_mat[1, 1] = -1000.002
    mig_mat[1, 2] = 0.001
    mig_mat[1, 3] = 0.001
    mig_mat[2, 0] = 0.001
    mig_mat[2, 1] = 0.001
    mig_mat[2, 2] = -1000.002
    mig_mat[2, 3] = 1000
    mig_mat[3, 0] = 0.001
    mig_mat[3, 1] = 0.001
    mig_mat[3, 2] = 1000
    mig_mat[3, 3] = -1000.002

    m, v = wfmoments.build_arbitrary(1e-3, mig_mat)
    eq = wfmoments.compute_equilibrium(m, v)
    assert np.isclose(
        wfmoments.compute_fst_nei(eq, [0], [2]),
        wfmoments.compute_fst_nei(eq, [0, 1], [2, 3])
    )
    assert np.isclose(
        wfmoments.compute_fst_nei(eq, [0], [1]), 0, atol=1e-3
    )
    assert np.isclose(
        wfmoments.compute_fst_nei(eq, [0], [2]),
        wfmoments.compute_fst_nei(
            eq, [0, 1], [2, 3], [0.25, 0.75], [0.75, 0.25]
        )
    )
    assert np.isclose(
        wfmoments.compute_fst_nei(eq, [0], [1], [100], [100]), 0, atol=1e-3
    )


def test_compute_fst_hudson():
    m, v = wfmoments.build_1d_spatial(1e-3, 1e-3, 2)
    eq = wfmoments.compute_equilibrium(m, v)
    fst_check = wfmoments.compute_fst_hudson(eq, [0], [1])
    pi_w = wfmoments.compute_pi(eq, [0])
    pi_t = wfmoments.compute_pi(eq, [0, 1])
    pi_b = 2 * pi_t - pi_w
    assert np.isclose(
        fst_check, 1 - pi_w / pi_b
    )

    fst_check_weight = wfmoments.compute_fst_hudson(
        eq, [0], [1], [10], [0.333]
    )
    assert np.isclose(fst_check_weight, 1 - pi_w / pi_b)

    mig_mat = np.zeros((4, 4))
    mig_mat[0, 0] = -1000.002
    mig_mat[0, 1] = 1000
    mig_mat[0, 2] = 0.001
    mig_mat[0, 3] = 0.001
    mig_mat[1, 0] = 1000
    mig_mat[1, 1] = -1000.002
    mig_mat[1, 2] = 0.001
    mig_mat[1, 3] = 0.001
    mig_mat[2, 0] = 0.001
    mig_mat[2, 1] = 0.001
    mig_mat[2, 2] = -1000.002
    mig_mat[2, 3] = 1000
    mig_mat[3, 0] = 0.001
    mig_mat[3, 1] = 0.001
    mig_mat[3, 2] = 1000
    mig_mat[3, 3] = -1000.002

    m, v = wfmoments.build_arbitrary(1e-3, mig_mat)
    eq = wfmoments.compute_equilibrium(m, v)
    assert np.isclose(
        wfmoments.compute_fst_hudson(eq, [0], [2]),
        wfmoments.compute_fst_hudson(eq, [0, 1], [2, 3])
    )
    assert np.isclose(
        wfmoments.compute_fst_hudson(eq, [0], [1]), 0, atol=1e-3
    )

    m, v = wfmoments.build_1d_spatial(1e-3, 1e-3, 10)
    eq = wfmoments.compute_equilibrium(m, v)
    fst_check = wfmoments.compute_fst_hudson(eq, range(5), range(5, 10))
    pi_w = (0.5 * wfmoments.compute_pi(eq, range(5))
            + 0.5 * wfmoments.compute_pi(eq, range(5, 10)))
    pi_t = wfmoments.compute_pi(eq, range(10))
    pi_b = 2 * pi_t - pi_w
    assert np.isclose(
        fst_check, 1 - pi_w / pi_b
    )

    w1 = np.random.random(5)
    w2 = np.random.random(5)
    w1 /= w1.sum()
    w2 /= w2.sum()
    fst_check_weight = wfmoments.compute_fst_hudson(
        eq, range(5), range(5, 10), w1, w2
    )
    pi_w = (0.5 * wfmoments.compute_pi(eq, range(5), w1)
            + 0.5 * wfmoments.compute_pi(eq, range(5, 10), w2))
    pi_t = wfmoments.compute_pi(eq, range(10), list(w1) + list(w2))
    pi_b = 2*pi_t - pi_w
    assert np.isclose(
        fst_check_weight, 1 - pi_w / pi_b
    )


def test_reseed():
    m, v = wfmoments.build_1d_spatial(1e-3, 1e-3, 2)
    eq = wfmoments.compute_equilibrium(m, v)
    old_pi = wfmoments.compute_pi(eq, [0])
    reseeded = wfmoments.reseed([[1]], [[0]], eq)
    assert reseeded[0] == reseeded[2]
    assert np.isclose(reseeded[1], reseeded[0])
    new_pi = wfmoments.compute_pi(reseeded, [0])
    assert np.isclose(old_pi, new_pi)
    new_pi = wfmoments.compute_pi(reseeded, [1])
    assert np.isclose(old_pi, new_pi)
    new_pi = wfmoments.compute_pi(reseeded, [0, 1])
    assert np.isclose(old_pi, new_pi)

    m, v = wfmoments.build_1d_spatial(1e-3, 1e-3, 4)
    eq = wfmoments.compute_equilibrium(m, v)
    old_pi = wfmoments.compute_pi(eq, [2, 3])
    reseeded = wfmoments.reseed([[0], [1]], [[2], [3]], eq)
    assert np.isclose(reseeded[0], eq[7])
    assert np.isclose(reseeded[4], eq[9])
    assert np.isclose(reseeded[1], eq[8])
    new_pi = wfmoments.compute_pi(reseeded, [0, 1])
    assert np.isclose(old_pi, new_pi)
    new_pi = wfmoments.compute_pi(reseeded, [2, 3])
    assert np.isclose(old_pi, new_pi)
    new_pi = wfmoments.compute_pi(reseeded, [0, 1, 2, 3])
    assert np.isclose(old_pi, new_pi)


def test_get_moments():
    m, v = wfmoments.build_1d_spatial(1e-3, 1e-3, 2)
    eq = wfmoments.compute_equilibrium(m, v)
    assert len(eq) == 3
    new_moments = wfmoments.get_moments(eq, [0])
    assert len(new_moments) == 1
    assert new_moments[0] == eq[0]
    new_moments = wfmoments.get_moments(eq, [1])
    assert len(new_moments) == 1
    assert new_moments[0] == eq[2]
    new_moments = wfmoments.get_moments(eq, [0, 1])
    assert np.allclose(new_moments, eq)
    new_moments = wfmoments.get_moments(eq, [1, 0])
    assert np.allclose(new_moments[::-1], eq)
    new_moments = wfmoments.get_moments(eq, [None, 0, 1])
    assert np.all(np.isnan(new_moments[0:3]))
    assert np.allclose(new_moments[3:], eq)


def test_get_moments_2d():
    m, v = wfmoments.build_2d_spatial(1e-3, 1e-3, 1, 10)
    moments = wfmoments.compute_equilibrium(m, v)

    truth = wfmoments.get_moments(moments, list(range(9)))
    extinction = np.zeros((1, 10), dtype=bool)
    extinction[0, 9] = True
    check = wfmoments.get_moments_2d(
        moments, None, extinction
    )
    assert np.allclose(truth, check)

    truth = wfmoments.get_moments(moments, list(range(8)))
    second_extinction = np.zeros((1, 10), dtype=bool)
    second_extinction[0, 9] = True
    second_extinction[0, 8] = True
    check = wfmoments.get_moments_2d(
        check, extinction, second_extinction
    )
    assert np.allclose(truth, check)

    truth = wfmoments.get_moments(moments, [0, 5])
    extinction = np.ones((1, 10), dtype=bool)
    extinction[0, 0] = False
    extinction[0, 5] = False
    check = wfmoments.get_moments_2d(
        moments, None, extinction
    )
    assert np.allclose(truth, check)

    extinction = np.ones((1, 10), dtype=bool)
    extinction[0, 0] = False
    extinction[0, 3] = False
    extinction[0, 5] = False
    step = wfmoments.get_moments_2d(
        moments, None, extinction
    )
    new_extinction = np.copy(extinction)
    new_extinction[0, 3] = True
    check = wfmoments.get_moments_2d(
        step, extinction, new_extinction
    )
    assert np.allclose(truth, check)

    extinction = np.ones((3, 3), dtype=bool)
    extinction[0, 0] = False
    m, v = wfmoments.build_2d_spatial(
        1e-3, 1e-3, 3, 3, extinct_demes=extinction
    )
    moments = wfmoments.compute_equilibrium(m, v)
    assert len(moments) == 1
    new_moments = wfmoments.get_moments_2d(
        moments, extinction, None
    )
    assert new_moments[0] == moments[0]
    assert np.all(np.isnan(new_moments[1:]))
