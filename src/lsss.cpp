#include "prettyprint.hpp"
#include "lsss.h"

int cross(point o, point a, point b) {
    return (a.first - o.first) * (b.second - o.second) - (a.second - o.second) * (b.first - o.first);
}

void merge_lch(std::vector<point> &a, const std::vector<point> b) {
    // Merge sorted point list b into sorted point list a, retaining only the lower convex hull
}

class uniq_pt_lst : public std::vector<point> {
    private:
        std::vector<point> tmp1, tmp2;
    public:
    void inc(const int d) {
        for (auto& p : *this) p.second += d;
    }
    void merge_lch(const uniq_pt_lst &other) { merge_lch(other, 0, 0); }
    void merge_lch(const uniq_pt_lst &other, const int r, const int d) {
        tmp1.resize(other.size());
        std::transform(other.begin(), other.end(), tmp1.begin(),
            [r, d](point p) {
                return std::make_pair<int,int>(p.first + r, p.second + d);
            }
        );
        tmp2.resize(size() + tmp1.size());
        iterator it = std::set_union(begin(), end(), tmp1.begin(), tmp1.end(), tmp2.begin(), 
                [](point p1, point p2) { return (p1.first == p2.first) ? p1.second < p2.second : p1.first < p2.first; }
        );
        tmp2.resize(it - tmp2.begin());
        clear();
        for (point p : tmp2) {
            if (size() && p.second > back().second) continue;
            while (size() >= 2 && cross(end()[-2], end()[-1], p) <= 0)
                pop_back();
            push_back(p);
        }
    }
};

std::vector<point> partition_haploid(const uint8_t* h, const uint8_t* H, const int L, const int N, const int n_threads) {
    uniq_pt_lst V;
    V.emplace_back(0,0);
    std::vector<uniq_pt_lst> V_n(N, V);
    std::vector<int> Nr;
    ThreadPool pool(n_threads);
    std::vector<std::future<void> > res;
    for (int ell = 0; ell < L; ++ell) {
       res.clear();
       for (int n = 0; n < N; ++n) {
           res.push_back(pool.enqueue([&V_n, &V, H, N, h, ell] (const int n) {
               int m = h[ell] != H[ell * N + n];
               V_n.at(n).inc(m);
               V_n.at(n).merge_lch(V, 1, m);
           }, n));
       }
       for (auto &&r : res) r.get();
       V.clear();
       for (int n = 0; n < N; ++n)
           V.merge_lch(V_n.at(n));
    }
    return V;
}

std::vector<point> partition_diploid(const uint8_t* g, const uint8_t* H, const int L, const int N, const int n_threads) {
    // g: L x 2
    // H: L x N
    uniq_pt_lst V;
    V.emplace_back(0,0);
    std::vector<uniq_pt_lst> V_n(N, V), new_V_n(N);
    std::vector<std::vector<uniq_pt_lst> > V_n_n(N, std::vector<uniq_pt_lst>(N, V));
    ThreadPool pool(n_threads);
    std::vector<std::future<void> > res;
    for (int ell = 0; ell < L; ++ell) {
       int g1 = g[2 * ell];
       int g2 = g[2 * ell + 1];
       res.clear();
       for (auto &v : new_V_n) v.clear();
       for (int n1 = 0; n1 < N; ++n1) {
           res.push_back(pool.enqueue([ell, g1, g2, H, N, &new_V_n, &V, &V_n, &V_n_n] (const int n1) {
               for (int n2 = 0; n2 < N; ++n2) {
                   int h1 = H[ell * N + n1];
                   int h2 = H[ell * N + n2];
                   int m = std::min((h1 != g1) + (h2 != g2), (h2 != g1) + (h1 != g2));
                   uniq_pt_lst &v = V_n_n.at(n1).at(n2);
                   v.inc(m);
                   v.merge_lch(V_n.at(n1), 1, m);
                   v.merge_lch(V_n.at(n2), 1, m);
                   v.merge_lch(V, 2, m);
                   new_V_n.at(n1).merge_lch(v);
               }
           },n1));
       }
       for (auto &&r : res) r.get();
       V_n = new_V_n;
       V.clear();
       for (uniq_pt_lst &v : V_n) V.merge_lch(v);
   }
    return V;
}

#include <time.h>
#include <stdlib.h>

double runif() { return ((double)rand() / (double)RAND_MAX); }

int main(int argc, char** argv) {
/*
    srand(time(NULL));
    const int N = 300;
    const int L = 500;
    int H[N*L];
    int g[2*L];
    for (int i = 0; i < N * L; ++i) H[i] = runif() < .5;
    for (int i = 0; i < 2 * L; ++i) g[i] = runif() < .5;
    std::cout << partition_diploid(g, H, N, L, 10) << std::endl;
*/
    uint8_t H[][4] = {{1,1,0,0},{0,0,1,1},{1,0,0,1},{1,1,1,1},{0,0,0,0}};
    uint8_t g[5][2] = {{0,0}, {1,1}, {1,0}, {1,0}, {0,0}};
    std::cout << partition_diploid(&g[0][0], &H[0][0], 5, 4, 10) << std::endl;
}