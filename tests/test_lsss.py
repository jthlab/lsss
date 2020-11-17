import pytest
import numpy as np
from lsss.lsss import LiStephensSurface
import msprime as msp


@pytest.fixture
def rnd_ts():
    return msp.simulate(
        sample_size=10, length=1e5, recombination_rate=1e-4, mutation_rate=1e-4
    )


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
    beta = ls.run()
    L, N = ls.H.shape
    assert len(beta) == 2
    assert beta[0] == (0, 0)
    assert beta[1] == (L, 0)


def test_from_ts_dip_perfect_match_simple():
    ts = msp.simulate(
        sample_size=10, length=10, mutation_rate=0.1, recombination_rate=0.1
    )
    ls = LiStephensSurface.from_ts(ts, [0, 1], np.arange(10))
    beta = ls.run()
    L, _ = ls.H.shape
    assert len(beta) == 2
    assert beta[0] == (0, 0)
    assert beta[1] == (2 * L, 0)


def test_from_ts_dip_perfect_match(rnd_ts):
    n = np.arange(rnd_ts.get_sample_size())
    ls = LiStephensSurface.from_ts(rnd_ts, n[:2], n)
    beta = ls.run()
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
    beta = ls.run()
    assert beta == [(0, 26), (1, 20), (2, 16), (3, 13), (4, 11), (88, 11)]


def test_known_dip_1():
    ts = msp.simulate(
        sample_size=10,
        length=1e4,
        mutation_rate=1e-2,
        recombination_rate=1e-2,
        random_seed=1,
    )
    G = ts.genotype_matrix()
    assert G.shape == (1082, 10)
    n = np.arange(10)
    ls = LiStephensSurface.from_ts(ts, n[-2:], n[:-2])
    assert ls.gh.shape == (1082, 2)
    assert ls.H.shape == (1082, 8)
    beta = ls.run()
    assert beta == [(0, 2), (2, 1), (2164, 1)]


def test_known_dip_2():
    H = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0]])
    g = np.array([[0, 0], [1, 1], [1, 0], [1, 0], [0, 0]])
    ls = LiStephensSurface(g, H)
    assert ls.run() == [(0, 1), (10, 1)]
