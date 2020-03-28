#pragma once

#include <armadillo>
#include <src/utils.hpp>
#include <src/tensor.hpp>

namespace nn_framework::scoring {
    template<typename T>
    double one_hot_accuracy_score(const Tensor<T> &y_true, const Tensor<T> &y_pred) {
        ensure(y_true.D == y_pred.D);
        ensure(y_true.Rank() == 2);
        arma::ucolvec y_true_vals = arma::index_max(y_true.Values(), 1);
        arma::ucolvec y_pred_vals = arma::index_max(y_pred.Values(), 1);
        arma::s64 cnt = 0;
        for (int i = 0; i < y_true.D[0]; i++) {
            if (y_true_vals.at(i) == y_pred_vals.at(i)) {
                cnt++;
            }
        }
        return static_cast<double>(cnt) / y_true.D[0];
    }

    template<typename T>
    double mse_score(const Tensor<T> &y_true, const Tensor<T> &y_pred) {
        ensure(y_true.D == y_pred.D);
        arma::Row<T> delta = (y_true.Values().as_row() - y_pred.Values().as_row());
        return arma::sum(delta % delta) / delta.n_elem;
    }
}