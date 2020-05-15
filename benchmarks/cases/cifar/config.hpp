#pragma once

#include <benchmarks/infrastructure/config_interface.hpp>

#include <src/scoring/scoring.hpp>
#include <src/neural_network.hpp>
#include <src/optimizer.hpp>
#include <src/tensor.hpp>
#include <src/layers/dense.hpp>
#include <src/layers/flatten.hpp>
#include <src/layers/convolution2d.hpp>
#include <src/layers/max_pooling2d.hpp>

namespace benchmarks {
    class CifarConfig : public Config {
    public:
        DataType LoadData(const std::string &data_path) override {
            std::vector<std::string> train_paths;
            for (int i = 1; i <= 5; i++) {
                train_paths.push_back(data_path + "/cifar-10-batches-bin/data_batch_" + std::to_string(i) + ".bin");
            }
            auto [x_train, y_train] = DoLoadData(train_paths);
            auto [x_test, y_test] = DoLoadData({data_path + "/cifar-10-batches-bin/test_batch.bin"});

            return {x_train, y_train, x_test, y_test};
        }

        NeuralNetwork<float> BuildModel() override {
            auto model = NeuralNetwork<float>(std::make_unique<RMSPropOptimizer<float>>(),
                                               std::make_unique<CategoricalCrossEntropyLoss<float>>());

            model
                    .AddLayer<Convolution2dLayer>(3, 3, 3, 3, ConvolutionPadding::Same, std::make_unique<GlorotUniformInitializer<float>>())
                    .AddLayer<ReLUActivationLayer>()
                    .AddLayer<MaxPooling2dLayer>(2, 2)

                    .AddLayer<Convolution2dLayer>(3, 5, 3, 3, ConvolutionPadding::Same, std::make_unique<GlorotUniformInitializer<float>>())
                    .AddLayer<ReLUActivationLayer>()
                    .AddLayer<MaxPooling2dLayer>(2, 2)

                    .AddLayer<FlattenLayer>(std::vector<int>{0, 5, 8, 8})
                    .AddLayer<DenseLayer>(320, 100)
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
        static std::tuple<Tensor<float>, Tensor<float>> DoLoadData(const std::vector<std::string>& paths) {
            std::vector<unsigned char> raw_data;
            for (const auto& path : paths) {
                std::ifstream input(path, std::ios::binary);
                ensure(input.is_open(), "can't open file " + path);
                raw_data.insert(raw_data.end(), std::istreambuf_iterator<char>(input), {});
            }
            int row_size = 1 + 32 * 32 * 3; // label + image
            size_t row_count = raw_data.size() / row_size;
            std::vector<std::vector<int>> raw_data_x(row_count);
            std::vector<std::vector<int>> raw_data_y(row_count);
            for (size_t i = 0; i < raw_data.size(); i++) {
                if (i % row_size == 0) {
                    raw_data_y[i / row_size].push_back(raw_data[i]);
                } else {
                    raw_data_x[i / row_size].push_back(raw_data[i]);
                }
            }
            auto X = Tensor<int>::fromVector(raw_data_x);
            auto y = Tensor<int>::fromVector(raw_data_y);
            auto y_one_hot = nn_framework::data_processing::OneHotEncoding(y).ConvertTo<float>();
            auto X_float = X.ConvertTo<float>();
            auto X_norm = Tensor<float>{X_float.D, X_float.Values() / 255};

            auto reshaper = FlattenLayer<float>({X_norm.D[0], 3, 32, 32});
            auto X_reshaped = reshaper.PullGradientsBackward(Tensor<float>(), X_norm).input_gradients;

            ensure(X_reshaped.D == TensorDimensions({static_cast<int>(row_count), 3, 32, 32}));
            ensure(y_one_hot.D == TensorDimensions({static_cast<int>(row_count), 10}));

            return {X_reshaped, y_one_hot};
        }
    };
}