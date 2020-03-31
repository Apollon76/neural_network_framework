#pragma once

#include <armadillo>
#include <src/utils.hpp>
#include <src/tensor.hpp>

namespace nn_framework::data_processing {
    template<typename T>
    Tensor<T> OneHotEncoding(const Tensor<T> &matrix) {
        ensure(matrix.Rank() == 1 || (matrix.Rank() == 2 && matrix.D[1] == 1),
               "one hot available only for Nx1 matrices or vectors");
        auto max = matrix.Values().max();
        auto result = Tensor<T>::filled({matrix.D[0], max + 1}, arma::fill::zeros);
        for (int i = 0; i < matrix.D[0]; i++) {
            result.Values().row(i).col(matrix.Values().at(i, 0)) = 1;
        }
        return result;
    }

    template<class T>
    class Scaler {
    public:
        void Fit(const Tensor<T> &matrix) {
            ensure(matrix.Rank() == 2, "only 2-dimensional matrices are supported");
            auto values = matrix.Values();
            mean = arma::mean(values);
            stddev = arma::stddev(values);
            stddev.transform([](double val) { return val < eps ? 1 : val; });
            fitted = true;
        }

        Tensor<T> Transform(const Tensor<T> &matrix) const {
            ensure(fitted, "scaler should be fitted before using transform");
            ensure(matrix.Rank() == 2, "only 2-dimensional matrices are supported");
            auto values = matrix.Values();
            return Tensor<T>(
                    matrix.D,
                    (values - arma::ones(values.n_rows, 1) * mean) / (arma::ones(values.n_rows, 1) * stddev)
            );
        }

    private:
        bool fitted = false;
        arma::mat mean;
        arma::mat stddev;

        constexpr static double eps = 1e-10;
    };

    template<class T>
    class TrainTestSplitter {
    public:
        explicit TrainTestSplitter(int random_seed) :
                random_seed(random_seed) {
        }

        TrainTestSplitter() :
                random_seed(std::random_device()()) {
        }

        std::tuple<Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>>
        Split(const Tensor<T> &x, const Tensor<T> &y, double train_portion) {
            return Split(x, y, static_cast<size_t>(train_portion * x.D[0]));
        }

        std::tuple<Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>>
        Split(const Tensor<T> &xT, const Tensor<T> &yT, size_t train_size) {
            ensure(xT.Rank() == 2, "only 2-dimensional matrices are supported");
            ensure(yT.Rank() == 2, "only 2-dimensional matrices are supported");
            auto x = xT.Values();
            auto y = yT.Values();
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
            auto x_train = Tensor<T>::filled({(int)train_size, (int)x.n_cols}, arma::fill::zeros);
            auto y_train = Tensor<T>::filled({(int)train_size, (int)y.n_cols}, arma::fill::zeros);
            auto x_test = Tensor<T>::filled({(int)test_size, (int)x.n_cols}, arma::fill::zeros);
            auto y_test = Tensor<T>::filled({(int)test_size, (int)y.n_cols}, arma::fill::zeros);

            for (size_t i = 0; i < x.n_rows; i++) {
                if (i < train_size) {
                    x_train.Values().row(i) = x.row(inds[i]);
                    y_train.Values().row(i) = y.row(inds[i]);
                } else {
                    x_test.Values().row(i - train_size) = x.row(inds[i]);
                    y_test.Values().row(i - train_size) = y.row(inds[i]);
                }
            }

            return {x_train, y_train, x_test, y_test};
        }

    private:
        const int random_seed;
    };

    template <typename T>
    struct Data {
        Tensor<T> input;
        Tensor<T> output;
    };

    template<typename T>
    Data<T> ShuffleData(const Data<T>& data) {
        size_t sz = data.input.D[0];
        arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0, sz - 1, sz));

        auto new_input = data.input.Rows(ordering);
        auto new_output = data.output.Rows(ordering);

        return Data<T>{
                Tensor(data.input.D, new_input.Field()),
                Tensor(data.output.D, new_output.Field())
        };
    }

    template<typename T>
    std::vector<Data<T>> GenerateBatches(Data<T> data, size_t batch_size, bool shuffle) {
        if (shuffle) {
            data = ShuffleData<T>({data.input, data.output});
        }
        DLOG(INFO) << "data dim [input=" << FormatDimensions(data.input.D) << " "
                   << "output=" << FormatDimensions(data.output.D) << "]";
        std::vector<Data<T>> batches;
        size_t size = data.input.D[0];

        for (size_t i = 0; i < size; i += batch_size) {
            auto cur_batch_size = batch_size;
            if (i + batch_size >= size) {
                cur_batch_size = size - i;
            }
            arma::uvec rows = arma::linspace<arma::uvec>(i, i + cur_batch_size - 1, cur_batch_size);
            batches.push_back(Data<T>{data.input.Rows(rows), data.output.Rows(rows)});
        }
        return batches;
    }
}