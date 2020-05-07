#include <armadillo>
#include <iostream>
#include "utils.hpp"

using namespace std;

arma::mat makeGrid(arma::mat base, int rows, int cols) {
    auto grid = arma::mat(base.n_rows * rows, base.n_cols * cols, arma::fill::zeros);
    for (int i = 0; i < rows; i++) {
        for (int s = 0; s < cols; s++) {
            grid.submat(i * base.n_rows, s * base.n_cols, (i + 1) * base.n_rows - 1, (s + 1) * base.n_cols - 1) = base;
        }
    }
    return grid;
}

int main() {
    int n = 5000;
    int m = 10;
    auto x = arma::mat(n, n, arma::fill::randu);
    auto y = arma::mat(m, m, arma::fill::randu);
    {
        auto t = Timer("Armadillo");
        auto z = arma::conv2(x, y, "same");
        std::cout << arma::accu(z) << std::endl;
    }
    {
        auto t = Timer("Naive");
        auto z = arma::mat(n, n, arma::fill::zeros);
        for (int i = 0; i < n; i++) {
            for (int s = 0; s < n; s++) {
                auto subX = x.submat(i, s, min(i + m - 1, n - 1), min(s + m - 1, n - 1));
                if (i + m <= n && s + m <= n) {
                    z(i, s) = arma::accu(subX % y);
                } else {
                    auto subY = y.submat(0, 0, min(m - 1, n - i - 1), min(m - 1, n - s - 1));
                    z(i, s) = arma::accu(subX % subY);
                }
            }
        }
        std::cout << arma::accu(z) << std::endl;
    }
    {
        auto t = Timer("Naive grid");
        auto grid = makeGrid(y, 1, n / m);
        auto z = arma::mat(n, n, arma::fill::zeros);
        for (int i = 0; i < n; i++) {
            for (int s = 0; s < m; s++) {
                auto subX = x.submat(i, s, min(i + m - 1, n - 1), n - 1);
                auto subY = grid.submat(0, 0, min(m - 1, n - i - 1), n - s - 1);
                arma::mat p = subX % subY;
                for (int ns = s; ns < n; ns += m) {
                    z(i, ns) = arma::accu(p.submat(0, ns - s, min(m - 1, n - i - 1), min(ns - s + m - 1, n - s - 1)));
                }
            }
        }
        std::cout << arma::accu(z) << std::endl;
    }
    return 0;
}