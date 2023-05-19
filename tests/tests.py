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

    curr_pi = pi
    curr_theta = 1e-1
    for i in range(10):
        curr_theta *= 2
        m, v = wfmoments.build_1d_spatial(curr_theta, 1e-3, 10)
        eq = wfmoments.compute_equilibrium(m, v)
        pi = wfmoments.compute_pi(eq, [0, 1, 9])
        assert pi > curr_pi
        curr_pi = pi


def test_reseed():
    m, v = wfmoments.build_1d_spatial(1e-3, 1e-3, 2)
    eq = wfmoments.compute_equilibrium(m, v)
    old_pi = wfmoments.compute_pi(eq, [0])
    reseeded = wfmoments.reseed([1], [0], eq)
    assert reseeded[0] == reseeded[2]
    assert np.isclose(reseeded[1], reseeded[0])
    new_pi = wfmoments.compute_pi(reseeded, [0])
    assert np.isclose(old_pi, new_pi)
    new_pi = wfmoments.compute_pi(reseeded, [1])
    assert np.isclose(old_pi, new_pi)
    new_pi = wfmoments.compute_pi(reseeded, [0, 1])
    assert np.isclose(old_pi, new_pi)
