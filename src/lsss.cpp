#include "lsss.h"

int cross(point o, point a, point b) {
    return (a.first - o.first) * (b.second - o.second) - (a.second - o.second) * (b.first - o.first);
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
        std::transform(other.begin(), other.end(), tmp1.begin(), [r, d](point p) {return std::make_pair<int,int>(p.first + r, p.second + d);});
        tmp2.resize(size() + tmp1.size());
        iterator it = std::set_union(begin(), end(), tmp1.begin(), tmp1.end(), tmp2.begin(), 
                [](point p1, point p2) { return (p1.first == p2.first) ? p1.second < p2.second : p1.first < p2.first; }
        );
        tmp2.resize(it - tmp2.begin());
        clear();
        int last_x = -1;
        it = tmp2.begin();
        while (it != tmp2.end()) {
            if (last_x == (*it).first) {
                it++;
            } else {
                last_x = (*it).first;
                while (size() >= 2 && cross(end()[-2], end()[-1], *it) <= 0)
                    pop_back();
                push_back(*it++);
            }
        }
    }
};

std::vector<point> partition_haploid(const int* h, const int* H, const int L, const int N, const int n_threads) {
    int ell, n;
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

std::vector<point> partition_diploid(const int* g, const int* H, const int L, const int N, const int n_threads) {
    // g: L x 2
    // H: L x N
    int ell, g1, g2;
    uniq_pt_lst V;
    V.emplace_back(0,0);
    std::vector<uniq_pt_lst> V_n(N, V), new_V_n(N);
    std::vector<std::vector<uniq_pt_lst> > V_n_n(N, std::vector<uniq_pt_lst>(N, V));
    ThreadPool pool(n_threads);
    std::vector<std::future<void> > res;
    for (int ell = 0; ell < L; ++ell) {
       g1 = g[2 * ell];
       g2 = g[2 * ell + 1];
       res.clear();
       for (int n1 = 0; n1 < N; ++n1) {
           res.push_back(pool.enqueue([ell, g1, g2, H, N, &new_V_n, &V, &V_n, &V_n_n] (const int n1) {
               new_V_n.at(n1).clear();
               for (int n2 = n1; n2 < N; ++n2) {
                   int h1 = H[ell * N + n1];
                   int h2 = H[ell * N + n2];
                   int m = ((h1 != g1) + (h1 != g2) + (h2 != g1) + (h2 != g2)) / 2;
                   V_n_n.at(n1).at(n2).inc(m);
                   V_n_n.at(n1).at(n2).merge_lch(V_n.at(n1), 1, m);
                   V_n_n.at(n1).at(n2).merge_lch(V_n.at(n2), 1, m);
                   V_n_n.at(n1).at(n2).merge_lch(V, 2, m);
                   new_V_n.at(n1).merge_lch(V_n_n.at(n1).at(n2));
               }
           }, n1));
       }
       for (auto &&r : res) r.get();
       V_n = new_V_n;
       V.clear();
       for (int n1 = 0; n1 < N; ++n1)
           V.merge_lch(V_n.at(n1));
    }
    return V;
}

#include <time.h>
#include <stdlib.h>
#include "prettyprint.hpp"

double runif() { return ((double)rand() / (double)RAND_MAX); }

int main(int argc, char** argv) {
    srand(time(NULL));
    const int N = 300;
    const int L = 500;
    int H[N*L];
    int g[2*L];
    for (int i = 0; i < N * L; ++i) H[i] = runif() < .5;
    for (int i = 0; i < 2 * L; ++i) g[i] = runif() < .5;
    std::cout << partition_diploid(g, H, N, L, 10) << std::endl;
}