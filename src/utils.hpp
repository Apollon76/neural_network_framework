#pragma once

#include <armadillo>
#include <vector>

arma::mat CreateMatrix(const std::vector<std::vector<double>> &values) {
    auto mat = arma::mat(values.size(), values[0].size());
    for (int i = 0; i < (int) values.size(); i++) {
        for (int s = 0; s < (int) values[0].size(); s++) {
            mat.at(i, s) = values[i][s];
        }
    }
    return mat;
}
