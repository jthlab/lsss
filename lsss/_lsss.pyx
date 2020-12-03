from typing import List, Tuple

import numpy as np

from libc.stdint cimport uint8_t
from libcpp.pair cimport pair
from libcpp.vector cimport vector


cdef extern from "lsss.h":
    ctypedef pair[int, int] point
    vector[point] partition_diploid(const uint8_t* g, const uint8_t* H, const int N, const int L, const int num_threads)
    vector[point] partition_haploid(const uint8_t* h, const uint8_t* H, const int N, const int L, const int num_threads)

def partition_ls(gh: np.ndarray, uint8_t[:, ::1] H, int num_threads) -> List[Tuple[int, int]]:
    L = H.shape[0]
    N = H.shape[1]
    cdef uint8_t[:] h_
    cdef uint8_t[:, ::1] g_
    if gh.ndim == 1:
        h_ = gh
        assert H.shape[0] == L
        return partition_haploid(&h_[0], &H[0, 0], L, N, num_threads)
    else:
        g_ = gh
        assert gh.ndim == 2
        assert gh.shape[0] == L
        assert gh.shape[1] == 2
        return partition_diploid(&g_[0, 0], &H[0, 0], L, N, num_threads)