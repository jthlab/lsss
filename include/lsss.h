#pragma once

#include <algorithm>
#include <iterator>
#include <vector>
#include <utility>
#include <map>
#include <iostream>

#include "ThreadPool.h"

typedef std::pair<int, int> point;
std::vector<point> partition_diploid(const int* g, const int* H, const int L, const int N, const int n_threads);
std::vector<point> partition_haploid(const int* h, const int* H, const int L, const int N);
