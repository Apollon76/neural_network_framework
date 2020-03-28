#pragma once

#include <armadillo>
#include <src/utils.hpp>
#include <src/tensor.hpp>

namespace nn_framework::data_processing {
    Tensor<int> OneHotEncoding(const Tensor<int> &matrix) {
        ensure(matrix.Rank() == 1 || (matrix.Rank() == 2 && matrix.D[1] == 1),
               "one hot available only for Nx1 matrices or vectors");
        auto max = matrix.Values().max();
        auto result = Tensor<int>::filled({matrix.D[0], max + 1}, arma::fill::zeros);
        for (int i = 0; i < matrix.D[0]; i++) {
            result.Values().row(i).col(matrix.Values().at(i, 0)) = 1;
        }
        return result;
    }
}