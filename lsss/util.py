import functools
from typing import Tuple

import numpy as np
from scipy.interpolate import PPoly


def ls_hap(ts, focal, panel, beta, alpha=1.0):
    G = ts.genotype_matrix()
    D = (G[:, [focal]] != G[:, panel]).astype(int)
    L, N = D.shape
    cost = np.zeros(N)
    m, r = np.zeros([2, N], int)
    for ell in range(L):
        F_t = cost.min() + beta
        r = np.where(F_t < cost, 1 + r[cost.argmin()], r)
        m = D[ell] + np.where(F_t < cost, m[cost.argmin()], m)
        cost = alpha * D[ell] + np.minimum(cost, F_t)
    n_star = cost.argmin()
    np.testing.assert_allclose(cost[n_star], alpha * m[n_star] + beta * r[n_star])
    return {"c": cost[n_star], "r": r[n_star], "m": m[n_star]}


def ls_dip(ts, focal, panel, beta, alpha=1.0):
    G = ts.genotype_matrix()
    g = G[:, focal].sum(1)
    G = G[:, panel]
    D = abs(G[:, :, None] + G[:, None, :] - g[:, None, None])
    L, N, _ = D.shape
    cost = np.zeros([N, N])
    m, r = np.zeros([2, N, N], int)
    for ell in range(L):
        # cost'[h1, h2] = D[ell, h1, h2] + min(cost[h1, h2], beta + cost[h1, :].min(), beta + cost[:, h2].min(), 2 * beta + cost.min())
        c1 = beta + cost.min(axis=0, keepdims=True)
        c2 = beta + cost.min(axis=1, keepdims=True)
        c3 = 2 * beta + cost.min(keepdims=True)
        m_star = functools.reduce(np.minimum, [c1, c2, c3, cost])
        switch = [m_star == c1, m_star == c2, m_star == c3, m_star == cost]
        r = np.select(
            switch,
            [
                1 + r[cost.argmin(axis=0), np.arange(N)][None],
                1 + r[np.arange(N), cost.argmin(axis=1)][:, None],
                2 + r.flat[cost.argmin()][None, None],
                r,
            ],
        )
        m = D[ell] + np.select(
            switch,
            [
                m[cost.argmin(axis=0), np.arange(N)][None],
                m[np.arange(N), cost.argmin(axis=1)][:, None],
                m.flat[cost.argmin()][None, None],
                m,
            ],
        )
        cost = alpha * D[ell] + m_star
    n_star = cost.argmin()
    np.testing.assert_allclose(
        cost.flat[n_star], alpha * m.flat[n_star] + beta * r.flat[n_star]
    )
    return {"c": cost.min(), "r": r.flat[n_star], "m": m.flat[n_star]}


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
