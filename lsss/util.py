import functools
from typing import Tuple

import numpy as np
from scipy.interpolate import PPoly


def ls_hap(ts_or_G, focal, panel, beta, alpha=1.0):
    try:
        G = ts_or_G.genotype_matrix()
    except AttributeError:
        G = ts_or_G
    D = (G[:, [focal]] != G[:, panel]).astype(int)
    L, N = D.shape
    cost = np.zeros(N)
    m, r = np.zeros([2, N], int)
    ibd = np.zeros(N, int)
    c_star = np.zeros([L, 2], int)  # the lowest cost haplotype and its ibd tract length
    n_star = 0
    for ell in range(L):
        F_t = cost.min() + beta
        recomb = F_t < cost
        ibd += 1
        ibd[recomb] = 1
        r[recomb] = 1 + r[n_star]
        m = D[ell] + np.where(recomb, m[n_star], m)
        cost = alpha * D[ell] + np.minimum(cost, F_t)
        n_star = cost.argmin()
        c_star[ell] = [n_star, ibd[n_star]]
    np.testing.assert_allclose(cost[n_star], alpha * m[n_star] + beta * r[n_star])
    # backtrack to find the optimal copying path
    path = _backtrace(c_star)
    assert len(path) - 1 == r[n_star]
    assert path[:, 1].sum() == L
    return {"c": cost[n_star], "r": r[n_star], "m": m[n_star], "path": path}


def _backtrace(c_star):
    ell = len(c_star) - 1
    path = []
    while ell >= 0:
        path.append(c_star[ell])
        ell -= path[-1][-1]
    path = np.array(path)[::-1]
    return path


def ls_dip(ts_or_G, focal, panel, beta, alpha=1.0):
    try:
        G = ts_or_G.genotype_matrix()
    except AttributeError:
        G = ts_or_G
    g = G[:, focal].sum(1)
    H = G[:, panel]
    D = abs(H[:, :, None] + H[:, None, :] - g[:, None, None])
    L, N, _ = D.shape
    NN = np.arange(N)
    cost = np.zeros([N, N])
    m, r = np.zeros([2, N, N], int)
    m_star, switch = _switch(beta, cost)
    # quantities needed for traceback
    ibd = np.zeros([N, N], int)
    c_star = np.zeros([L, 3], int)  # the lowest cost haplotypes
    n_star = [0, 0]
    for ell in range(L):
        # cost'[h1, h2] = D[ell, h1, h2] + min(cost[h1, h2],  # no recombination
        #                                      beta + cost[h1, :].min(),  # single recombination
        #                                      beta + cost[:, h2].min(),  # single recombination
        #                                      2 * beta + cost.min())  # double recombination
        # update ibd tract lengths
        ibd += 1
        ibd[~switch[3]] = 1
        r = np.select(
            switch,
            [
                1 + r[cost.argmin(axis=0), NN][None],
                1 + r[NN, cost.argmin(axis=1)][:, None],
                2 + r.flat[cost.argmin()][None, None],
                r,
            ],
        )
        m = D[ell] + np.select(
            switch,
            [
                m[cost.argmin(axis=0), NN][None],
                m[NN, cost.argmin(axis=1)][:, None],
                m.flat[cost.argmin()][None, None],
                m,
            ],
        )
        cost = alpha * D[ell] + m_star
        n_star = np.unravel_index(cost.argmin(), cost.shape)
        c_star[ell] = np.r_[n_star, ibd[n_star]]
        m_star, switch = _switch(beta, cost)
    np.testing.assert_allclose(cost[n_star], alpha * m[n_star] + beta * r[n_star])
    path = _backtrace(c_star)
    return {
        "c": cost.min(),
        "r": r[n_star],
        "m": m[n_star],
        "path": path,
        "g": g,
        "G": H,
        "alpha": alpha,
        "beta": beta,
    }


def _switch(beta, cost):
    c1 = beta + cost.min(axis=0, keepdims=True)
    c2 = beta + cost.min(axis=1, keepdims=True)
    c3 = 2 * beta + cost.min(keepdims=True)
    m_star = functools.reduce(np.minimum, [c1, c2, c3, cost])
    switch = [m_star == c1, m_star == c2, m_star == c3, m_star == cost]
    return m_star, switch


def beta_from_V(V):
    """Given a sorted list V=[(r_0, m_0), ..., (r_n, m_n)], return +oo=beta_0 > beta_1 > ...
    such that the optimal value function is constant on each beta interval.
    """
    p_K_i, J_K_i = np.transpose(V[::-1])
    betas = -np.diff(J_K_i) / np.diff(p_K_i)
    m = p_K_i[1:]
    c = np.r_[J_K_i[1], np.diff(betas) * m[:-1]].cumsum()
    return PPoly(x=np.append(betas, np.inf), c=[m, c])


def phase_function(s_beta: PPoly, N: int):
    """Given the s_beta function defined below, return a list [(t_i, m_i, r_i)] such that
    LS(theta, rho) = (m_i, r_i) for tan(expm1(theta) / expm1(rho)) \in [t_{i-1}, t_i).
    """

    def f(theta, rho):
        v = N * (np.expm1(theta) / 3) ** s_beta.x
        i = np.searchsorted(-v, -np.expm1(rho)) - 1
        assert v[i + 1] <= np.expm1(rho) < v[i]
        return s_beta.c[0, i]

    return f


def alpha_beta(theta: float, rho: float, n: int) -> Tuple[float, float]:
    p_theta = -np.expm1(-theta) / 3.0
    p_rho = -np.expm1(-rho) / n
    alpha = -np.log(p_theta) + np.log1p(-3 * p_theta)
    beta = -np.log(p_rho) + np.log1p(-n * p_rho)
    return alpha, beta
