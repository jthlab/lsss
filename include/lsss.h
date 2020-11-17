#pragma once

#include <stdint.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <utility>
#include <map>
#include <iostream>

#include "ThreadPool.h"

typedef std::pair<int, int> point;
std::vector<point> partition_diploid(const uint8_t* g, const uint8_t* H, const int L, const int N, const int n_threads);
std::vector<point> partition_haploid(const uint8_t* h, const uint8_t* H, const int L, const int N, const int n_threads);
