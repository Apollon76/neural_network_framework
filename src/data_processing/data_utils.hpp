#pragma once

#include <armadillo>
#include <src/utils.hpp>

namespace nn_framework::data_processing {
    arma::s32_mat OneHotEncoding(const arma::s32_mat& matrix) {
        ensure(matrix.n_cols == 1, "one hot available only for Nx1 matrices");
        auto max = matrix.max();
        arma::s32_mat result(matrix.n_rows, max + 1);
        for (size_t i = 0; i < matrix.n_rows; i++) {
            result.row(i).col(matrix.at(i, 0)) = 1;
        }
        return result;
    }
}