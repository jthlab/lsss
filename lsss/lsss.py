from dataclasses import dataclass
import os
from typing import List, Tuple, Union

import numpy as np

from . import _lsss


@dataclass
class LiStephensSurface:
    gh: np.ndarray
    H: np.ndarray

    def __post_init__(self):
        gh = self.gh = np.ascontiguousarray(self.gh, dtype=np.uint8)
        H = self.H = np.ascontiguousarray(self.H, dtype=np.uint8)
        if gh.shape[0] != H.shape[0]:
            raise ValueError("shape mismatch between gh and H")
        if gh.ndim == 2 and gh.shape[1] != 2:
            raise ValueError("genotypes shape should be Lx2")
        if 0 in H.shape:
            raise ValueError("empty genotype/haplotype array")

    def run(self, num_threads: int = os.cpu_count()):
        self._betas = _lsss.partition_ls(self.gh, self.H, num_threads)
        return self._betas

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
