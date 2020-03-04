#pragma once

#include <armadillo>
#include <src/utils.hpp>

namespace nn_framework::scoring {
    double one_hot_accuracy_score(arma::mat y_true, arma::mat y_pred) {
        ensure(y_true.n_rows == y_pred.n_rows);
        ensure(y_true.n_cols == y_pred.n_cols);
        arma::ucolvec y_true_vals = arma::index_max(y_true, 1);
        arma::ucolvec y_pred_vals = arma::index_max(y_pred, 1);
        arma::s64 cnt = 0;
        for (size_t i = 0; i < y_true.n_rows; i++) {
            if (y_true_vals.at(i) == y_pred_vals.at(i)) {
                cnt++;
            }
        }
        return static_cast<double>(cnt) / y_true.n_rows;
    }
}