#pragma once

#include <armadillo>
#include <vector>

template<typename T>
arma::Mat<T> CreateMatrix(const std::vector<std::vector<T>> &values) {
    auto mat = arma::Mat<T>(values.size(), values[0].size());
    for (int i = 0; i < (int) values.size(); i++) {
        for (int s = 0; s < (int) values[0].size(); s++) {
            mat.at(i, s) = values[i][s];
        }
    }
    return mat;
}

constexpr void ensure(bool value, const std::string& error) {
    if (!value) {
        throw std::runtime_error(error);
    }
}