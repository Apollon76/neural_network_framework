#pragma once

#include <armadillo>
#include <src/utils.hpp>

namespace nn_framework::data_processing {
    arma::s32_mat OneHotEncoding(const arma::s32_mat& matrix) {
        ensure(matrix.n_cols == 1, "one hot available only for Nx1 matrices");
        auto max = matrix.max();
        arma::s32_mat result(matrix.n_rows, max + 1, arma::fill::zeros);
        for (size_t i = 0; i < matrix.n_rows; i++) {
            result.row(i).col(matrix.at(i, 0)) = 1;
        }
        return result;
    }

    template <class T>
    class Scaler {
    public:
        void Fit(const arma::Mat<T>& matrix) {
            mean = arma::mean(matrix);
            stddev = arma::stddev(matrix);
            stddev.transform([](double val) { return val < eps ? 1 : val; });
            fitted = true;
        }

        arma::Mat<T> Transform(const arma::Mat<T>& matrix) const {
            ensure(fitted, "scaler should be fitted before using transform");
            return (matrix - arma::ones(matrix.n_rows, 1) * mean) / (arma::ones(matrix.n_rows, 1) * stddev);
        }

    private:
        bool fitted = false;
        arma::mat mean;
        arma::mat stddev;

        constexpr static double eps = 1e-10;
    };
}