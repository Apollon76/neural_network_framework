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

    template <class T>
    class TrainTestSplitter {
    public:
        explicit TrainTestSplitter(int random_seed) :
            random_seed(random_seed) {
        }

        TrainTestSplitter() :
            random_seed(std::random_device()()) {
        }

        std::tuple<arma::Mat<T>, arma::Mat<T>, arma::Mat<T>, arma::Mat<T>> Split(const arma::Mat<T>& x, const arma::Mat<T>& y, double train_portion) {
            return Split(x, y, static_cast<size_t>(train_portion * x.n_rows));
        }

        std::tuple<arma::Mat<T>, arma::Mat<T>, arma::Mat<T>, arma::Mat<T>> Split(const arma::Mat<T>& x, const arma::Mat<T>& y, size_t train_size) {
            ensure(x.n_rows == y.n_rows);
            ensure(0 < train_size);
            ensure(train_size <= x.n_rows);

            std::vector<int> inds(x.n_rows);
            for (size_t i = 0; i < inds.size(); i++) {
                inds[i] = i;
            }

            std::mt19937 generator(random_seed);
            std::shuffle(inds.begin(), inds.end(), generator);

            int test_size = x.n_rows - train_size;
            arma::Mat<T> x_train(train_size, x.n_cols);
            arma::Mat<T> y_train(train_size, y.n_cols);
            arma::Mat<T> x_test(test_size, x.n_cols);
            arma::Mat<T> y_test(test_size, y.n_cols);

            for (size_t i = 0; i < x.n_rows; i++) {
                if (i < train_size) {
                    x_train.row(i) = x.row(inds[i]);
                    y_train.row(i) = y.row(inds[i]);
                } else {
                    x_test.row(i - train_size) = x.row(inds[i]);
                    y_test.row(i - train_size) = y.row(inds[i]);
                }
            }

            return {x_train, y_train, x_test, y_test};
        }

    private:
        const int random_seed;
    };
}