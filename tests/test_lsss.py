import msprime as msp
import numpy as np
import pytest

from lsss.lsss import LiStephensSurface
from lsss.util import ls_hap, ls_dip, phase_function, alpha_beta


@pytest.fixture(scope="module", params=range(1, 10))
def seed(request):
    return request.param


@pytest.fixture
def rng(seed):
    return np.random.default_rng(seed)


@pytest.fixture
def alpha(rng):
    return rng.exponential()


@pytest.fixture
def beta(rng):
    return rng.exponential()


@pytest.fixture
def rnd_ts(seed):
    return msp.simulate(
        sample_size=10,
        length=1e5,
        recombination_rate=1e-4,
        mutation_rate=1e-4,
        random_seed=seed,
    )


@pytest.fixture
def rnd_hap_ls(rnd_ts):
    n = np.arange(rnd_ts.get_sample_size())
    return LiStephensSurface.from_ts(rnd_ts, n[0], n[1:])


def test_empty_seq():
    with pytest.raises(ValueError):
        LiStephensSurface([], [[]])


def test_dim_mismatch():
    with pytest.raises(ValueError):
        LiStephensSurface(np.ones(3), np.ones([4, 5]))


def test_genotype_wrong_dim():
    LiStephensSurface(np.ones([3, 2]), np.ones([3, 5]))
    with pytest.raises(ValueError):
        LiStephensSurface(np.ones([3, 3]), np.ones([3, 5]))


def test_from_ts_hap_perfect_match(rnd_ts):
    n = np.arange(rnd_ts.get_sample_size())
    ls = LiStephensSurface.from_ts(rnd_ts, n[0], n)
    beta = ls.V
    L, N = ls.H.shape
    assert len(beta) == 2
    assert beta[0] == (0, 0)
    assert beta[1] == (L, 0)


def test_from_ts_dip_perfect_match_simple():
    ts = msp.simulate(
        sample_size=10, length=10, mutation_rate=0.1, recombination_rate=0.1
    )
    ls = LiStephensSurface.from_ts(ts, [0, 1], np.arange(10))
    beta = ls.V
    L, _ = ls.H.shape
    assert len(beta) == 2
    assert beta[0] == (0, 0)
    assert beta[1] == (2 * L, 0)


def test_from_ts_dip_perfect_match(rnd_ts):
    n = np.arange(rnd_ts.get_sample_size())
    ls = LiStephensSurface.from_ts(rnd_ts, n[:2], n)
    beta = ls.V
    L, _ = ls.H.shape
    assert len(beta) == 2
    assert beta[0] == (0, 0)
    assert beta[1] == (2 * L, 0)


def test_from_ts_hap(rnd_ts):
    LiStephensSurface.from_ts(rnd_ts, 0, list(range(1, 10)))


def test_from_ts_dip(rnd_ts):
    LiStephensSurface.from_ts(rnd_ts, [0, 1], list(range(10)))


def test_known_hap_1():
    ts = msp.simulate(
        sample_size=10,
        length=1e3,
        mutation_rate=1e-2,
        recombination_rate=1e-2,
        random_seed=1,
    )
    n = np.arange(ts.get_sample_size())
    ls = LiStephensSurface.from_ts(ts, n[0], n[1:])
    beta = ls.V
    assert beta == [(0, 0), (88, 0)]


def test_known_dip_1():
    ts = msp.simulate(
        sample_size=10,
        length=1e4,
        mutation_rate=1e-2,
        recombination_rate=1e-2,
        random_seed=1,
    )
    G = ts.genotype_matrix()
    assert G.shape == (1011, 10)
    n = np.arange(10)
    ls = LiStephensSurface.from_ts(ts, n[-2:], n[:-2])
    assert ls.gh.shape == (1011, 2)
    assert ls.H.shape == (1011, 8)
    beta = ls.V
    assert beta == [
        (0, 271),
        (1, 223),
        (2, 184),
        (3, 156),
        (4, 138),
        (6, 106),
        (7, 92),
        (8, 81),
        (9, 75),
        (11, 64),
        (12, 60),
        (16, 48),
        (18, 44),
        (29, 33),
        (39, 28),
        (2022, 28),
    ]


def test_known_dip_2():
    H = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0]])
    g = np.array([[0, 0], [1, 1], [1, 0], [1, 0], [0, 0]])
    ls = LiStephensSurface(g, H)
    assert ls.V == [(0, 1), (10, 1)]


def test_true_invariant_hap(rnd_ts, alpha, beta):
    n = np.arange(rnd_ts.get_sample_size())
    focal = n[0]
    panel = n[1:]
    d1 = ls_hap(rnd_ts, focal, panel, alpha=alpha, beta=beta)
    d2 = ls_hap(rnd_ts, focal, panel, alpha=alpha / beta, beta=1.0)
    d3 = ls_hap(rnd_ts, focal, panel, alpha=1.0, beta=beta / alpha)
    np.testing.assert_allclose(
        *[[[u["m"], u["r"]], [v["m"], v["r"]]] for u, v in ((d1, d2), (d2, d3))]
    )


def _test_partition(ts, focal, panel, alpha):
    ls = LiStephensSurface.from_ts(ts, focal, panel)
    betas = ls.C_beta
    mid = (betas.x[:-1] + betas.x[1:]) / 2.0
    ls_true = ls_hap if np.size(focal) == 1 else ls_dip
    f = lambda b: ls_true(ts, focal, panel, beta=b, alpha=alpha)
    for b in np.r_[betas.x[:-1], mid[:-1], np.random.rand(100) * betas.x[-2] * 2]:
        d = f(b)
        c = b / alpha
        assert np.allclose(c * d["r"] + d["m"], betas(c))


def test_beta_partition_dip_theta1(rnd_ts):
    n = np.arange(rnd_ts.get_sample_size())
    focal = n[:2]
    panel = n[2:]
    _test_partition(rnd_ts, focal, panel, alpha=1.0)


def test_beta_partition_hap_general_theta(rnd_ts, alpha):
    n = np.arange(rnd_ts.get_sample_size())
    focal = n[0]
    panel = n[1:]
    _test_partition(rnd_ts, focal, panel, alpha=alpha)


def test_C_s_beta(rnd_hap_ls, rng):
    C = rnd_hap_ls.C_beta
    s = rnd_hap_ls.s_beta
    for beta in np.r_[0.0, rng.exponential(size=10)]:
        r, m = s(beta)
        assert np.allclose(m + beta * r, C(beta))


def test_phase_function_hap(rnd_ts, rng):
    N = rnd_ts.get_sample_size()
    n = np.arange(N)
    ls = LiStephensSurface.from_ts(rnd_ts, n[0], n[1:])
    f = phase_function(ls.s_beta, ls.N)
    for theta, rho in rng.exponential(size=(10, 2)):
        alpha, beta = alpha_beta(theta, rho, ls.N)
        if alpha < 0 or beta < 0:
            continue
        v = ls.N * (np.expm1(theta) / 3) ** ls.s_beta.x
        r, m = f(theta, rho)
        d = ls_hap(rnd_ts, n[0], n[1:], alpha=alpha, beta=beta)
        assert m == d["m"]
        assert r == d["r"]


def test_ls_hap_path_simple():
    G = np.array([[0, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]]).T
    d = ls_hap(G, 0, [1, 2], alpha=1.0, beta=0.5)
    assert d["m"] == 0
    assert d["r"] == 1
    assert np.all(d["path"] == [[0, 2], [1, 2]])


def test_ls_hap_path(rnd_ts, alpha, beta):
    G = rnd_ts.genotype_matrix()
    L, H = G.shape
    d = ls_hap(rnd_ts, 0, np.arange(1, H), alpha=alpha, beta=beta)
    focal = G[:, 0]
    panel = G[:, 1:]
    path = d["path"]
    assert len(path) - 1 == d["r"]
    assert path[:, 1].sum() == L
    m = 0
    ell = 0
    for h, k in path:
        m += (focal[ell : ell + k] != panel[ell : ell + k, h]).sum()
        ell += k
    assert m == d["m"]


def test_ls_dip_path_simple(alpha, beta):
    G = np.array(
        [[0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1]]
    ).T
    focal = G[:, :2].sum(1)
    panel = G[:, 2:]
    d = ls_dip(G, [0, 1], [2, 3, 4], alpha=alpha, beta=beta)
    _test_ls_dip_path(d)


def test_ls_dip_path_simple2(alpha, beta):
    G = np.array([[0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]).T
    d = ls_dip(G, [0, 1], [2, 3], alpha=alpha, beta=beta)
    _test_ls_dip_path(d)
    truth = np.array([[0, 1, 4]])
    assert (d["path"] == truth).all() or (d["path"] == truth[:, [1, 0, 2]]).all()


def test_ls_dip_path_simple3(alpha, beta):
    G = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]).T
    focal = G[:, :2].sum(1)
    panel = G[:, 2:]
    d = ls_dip(G, [0, 1], [2, 3], alpha=alpha, beta=beta)
    _test_ls_dip_path(d)


def test_ls_dip_path_simple5(alpha, beta):
    G = np.array([[0, 0], [1, 0], [0, 1], [1, 0]]).T  # [1, 0]
    focal = G[:, :2].sum(1)
    panel = G[:, 2:]
    d = ls_dip(G, [0, 1], [2, 3], alpha=alpha, beta=beta)
    _test_ls_dip_path(d)


def test_ls_dip_path(rnd_ts, alpha, beta):
    G = rnd_ts.genotype_matrix()
    L, H = G.shape
    d = ls_dip(G, [0, 1], np.arange(2, H), alpha=alpha, beta=beta)
    focal = G[:, :2].sum(1)
    panel = G[:, 2:]
    _test_ls_dip_path(d)


def _test_ls_dip_path(d):
    # check all posistions accounted for
    g = d["g"]
    G = d["G"]
    p = d["path"]
    assert p[:, 2].sum() == G.shape[0]
    # check recombinations sum up
    ps = p[:, :2]
    s = [min((a != b).sum(), (a != b[::-1]).sum()) for a, b in zip(ps[:-1], ps[1:])]
    r = sum(s)
    m = ell = 0
    for h1, h2, span in d["path"]:
        m += abs(g[ell : ell + span] - G[ell : ell + span, [h1, h2]].sum(1)).sum()
        ell += span
    assert d["alpha"] * m + d["beta"] * r >= d["c"]
