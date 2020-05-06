#include <benchmarks/infrastructure/config_interface.cpp>

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

        NeuralNetwork<double> BuildModel() override {
            auto model = NeuralNetwork<double>(std::make_unique<AdamOptimizer<double>>(),
                                               std::make_unique<CategoricalCrossEntropyLoss<double>>());

            model.AddLayer<DenseLayer>(784, 100)
                    .AddLayer<SigmoidActivationLayer>()
                    .AddLayer<DenseLayer>(100, 10)
                    .AddLayer<SoftmaxActivationLayer>();

            return model;
        }

        double GetScore(const Tensor<double> &y_true, const Tensor<double> &y_pred) override {
            return nn_framework::scoring::one_hot_accuracy_score(y_true, y_pred);
        }

        std::string GetScoreName() override {
            return "categorical_accuracy";
        }

    private:
        static std::tuple<Tensor<double>, Tensor<double>> DoLoadData(const std::string &path) {
            auto csv_data_provider = nn_framework::io::CsvReader(path, false);
            auto data = Tensor<int>::fromVector(csv_data_provider.LoadData<int>());
            auto X = Tensor<int>({data.D[0], data.D[1] - 1}, data.Values().tail_cols(data.D[1] - 1));
            auto y = Tensor<int>({data.D[0], 1}, data.Values().head_cols(1));
            auto y_one_hot = nn_framework::data_processing::OneHotEncoding(y);
            auto X_double = X.ConvertTo<double>();
            auto X_norm = Tensor<double>{X_double.D, X_double.Values() / 255};
            return {X_norm, y_one_hot.ConvertTo<double>()};
        }
    };
}