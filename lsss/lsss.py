import os
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
from scipy.interpolate import PPoly

from . import _lsss
from .util import beta_from_V, alpha_beta


@dataclass
class LiStephensSurface:
    """Compute the solution surface for the Li-Stephens model.

    Args:
        gh: Either a 1-dimensional integer array of length L, or a 2-dimensional integer array of shape [L, 2],
            representing focal haplotypes or genotypes, respectively.
        H: A two-dimensional integer array of shape [L, H] representing panel haplotypes.

    Notes:
        The integers in gh and H can represent any encoding of genotype information; it is only necessary to test for a
        allele equality in the L-S model. For example, in a biallelic model, the entries of gh and H are binary, while
        in a general tetra-allelic model, they can be in {0, 1, 2, 3}.
    """

    gh: np.ndarray
    H: np.ndarray
    num_threads: int = os.cpu_count()

    def __post_init__(self):
        gh = self.gh = np.ascontiguousarray(self.gh, dtype=np.uint8)
        H = self.H = np.ascontiguousarray(self.H, dtype=np.uint8)
        if gh.shape[0] != H.shape[0]:
            raise ValueError("shape mismatch between gh and H")
        if gh.ndim == 2 and gh.shape[1] != 2:
            raise ValueError("genotypes shape should be Lx2")
        if 0 in H.shape:
            raise ValueError("empty genotype/haplotype array")
        self.V = _lsss.partition_ls(self.gh, self.H, self.num_threads)

    @property
    def L(self):
        """Number of loci"""
        return self.H.shape[0]

    @property
    def N(self):
        """Number of haplotypes in panel"""
        return self.H.shape[1]

    @property
    def diploid(self) -> bool:
        """True if this is a diploid model"""
        return self.gh.ndim == 2

    @property
    def s_beta(self) -> PPoly:
        r""":math:`s(\beta)` with the property that

        .. math:: s(\beta) = (m(\pi^*),r(\pi^*)) \text{ where } \pi^* = \arg\min_\pi m(\pi) + \beta k(\pi).

        (See manuscript for notation.)
        """
        c = np.array(self.V[:-1][::-1])[None]
        return PPoly(x=self.C_beta.x, c=c)

    @property
    def C_beta(self) -> PPoly:
        r""":math:`C(\beta)` with the property that

        .. math:: C(\beta) = \min_\pi m(\pi) + \beta k(\pi).

        (See manuscript for notation.)
        """
        return beta_from_V(self.V)

    def __call__(self, theta, rho):
        # probability of mutation to any bp
        alpha, beta = alpha_beta(theta, rho, self.N)
        return self.s_beta(beta / alpha)

    def draw(self, ax=None):
        r"""Plot a phase diagram of solution space.

        Args:
            ax: A matplotlib axis on which to draw the plot, or matplotlib.pyplot.gca() if None.

        Notes:
            Assumes a tetraallelic model where the probability of a mutation between any two nucleotides
            :math:`X,Y \in \{A,C,G,T\}, X\neq Y` is

            .. math:: p_\theta = \frac{1 - e^{-\theta}}{3},

            where the population-scaled mutation rate is :math:`\theta`. Similarly, the probability of recombination
            onto another haplotype

            .. math:: \frac{1 - e^{-\rho}}{N},

            where :math:`N` is the size of the panel.

            FIXME
            See Figure XX in paper for an example of this plot.
        """
        if ax is None:
            try:
                import matplotlib.pyplot as plt
            except ImportError as e:
                raise ImportError("Plotting requires matplotlib") from e

            ax = plt.gca()

        # plot
        theta = np.geomspace(1.2, 1.6, 200)[:, None]
        expm1_rho = self.N * (np.expm1(theta) / 3) ** self.s_beta.x[None, :]
        ax.plot(theta, np.log1p(expm1_rho))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$\rho$")
        return ax

    @classmethod
    def from_ts(
        cls,
        ts: "tskit.TreeSequence",
        focal: Union[int, Tuple[int, int]],
        panel: List[int],
    ) -> "LiStephensSurface":
        G = ts.genotype_matrix()
        gh = G[:, focal].astype(np.int32)
        H = G[:, panel].astype(np.int32)
        return cls(gh, H)
