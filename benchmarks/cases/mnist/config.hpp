#pragma once

#include <benchmarks/infrastructure/config_interface.hpp>

#include <src/io/csv.hpp>
#include <src/scoring/scoring.hpp>
#include <src/neural_network.hpp>
#include <src/optimizer.hpp>
#include <src/tensor.hpp>
#include <src/layers/dense.hpp>

namespace benchmarks {
    class MnistConfig : public Config {
    public:
        DataType LoadData(const std::string &data_path) override {
            auto[x_train, y_train] = DoLoadData(data_path + "/train.csv");
            auto[x_test, y_test] = DoLoadData(data_path + "/test.csv");

            return {x_train, y_train, x_test, y_test};
        }

        NeuralNetwork<float> BuildModel() override {
            auto model = NeuralNetwork<float>(std::make_unique<AdamOptimizer<float>>(),
                                               std::make_unique<CategoricalCrossEntropyLoss<float>>());

            model.AddLayer<DenseLayer>(784, 100)
                    .AddLayer<SigmoidActivationLayer>()
                    .AddLayer<DenseLayer>(100, 10)
                    .AddLayer<SoftmaxActivationLayer>();

            return model;
        }

        double GetScore(const Tensor<float> &y_true, const Tensor<float> &y_pred) override {
            return nn_framework::scoring::one_hot_accuracy_score(y_true, y_pred);
        }

        std::string GetScoreName() override {
            return "categorical_accuracy";
        }

    private:
        static std::tuple<Tensor<float>, Tensor<float>> DoLoadData(const std::string &path) {
            auto csv_data_provider = nn_framework::io::CsvReader(path, false);
            auto data = Tensor<int>::fromVector(csv_data_provider.LoadData<int>());
            auto X = Tensor<int>({data.D[0], data.D[1] - 1}, data.Values().tail_cols(data.D[1] - 1));
            auto y = Tensor<int>({data.D[0], 1}, data.Values().head_cols(1));
            auto y_one_hot = nn_framework::data_processing::OneHotEncoding(y);
            auto X_float = X.ConvertTo<float>();
            auto X_norm = Tensor<float>{X_float.D, X_float.Values() / 255};
            return {X_norm, y_one_hot.ConvertTo<float>()};
        }
    };
}