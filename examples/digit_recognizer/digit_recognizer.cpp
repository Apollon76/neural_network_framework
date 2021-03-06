#include <memory>
#include <armadillo>
#include <glog/logging.h>
#include <cxxopts.hpp>
#include <src/io/csv.hpp>
#include <src/scoring/scoring.hpp>
#include <src/neural_network.hpp>
#include <src/layers/activations.hpp>
#include <src/optimizer.hpp>
#include <src/utils.hpp>
#include <src/layers/dense.hpp>
#include <src/callbacks/logging_callback.hpp>
#include <src/tensor.hpp>
#include <src/callbacks/progress_bar_callback.hpp>

std::tuple<Tensor<double>, Tensor<double>> LoadMnist(const std::string &path) {
    std::cout << "Loading mnist dataset from " << path << std::endl;
    Timer timer("Load of " + path, true);
    auto csv_data_provider = nn_framework::io::CsvReader(path, true);
    auto data = Tensor<int>::fromVector(csv_data_provider.LoadData<int>());
    auto X = Tensor<int>({data.D[0], data.D[1] - 1}, data.Values().tail_cols(data.D[1] - 1));
    auto y = Tensor<int>({data.D[0], 1}, data.Values().head_cols(1));
    auto y_one_hot = nn_framework::data_processing::OneHotEncoding(y);
    return {X.ConvertTo<double>(), y_one_hot.ConvertTo<double>()};
}

Tensor<double> LoadMnistX(const std::string &path) {
    Timer timer("Load of " + path, true);
    auto csv_data_provider = nn_framework::io::CsvReader(path, true);
    auto data = Tensor<int>::fromVector(csv_data_provider.LoadData<int>());
    return data.ConvertTo<double>();
}

NeuralNetwork<double> BuildMnistNN() {
    auto neural_network = NeuralNetwork<double>(std::make_unique<RMSPropOptimizer<double>>(0.01),
                                                std::make_unique<CategoricalCrossEntropyLoss<double>>());
    neural_network
            .AddLayer(std::make_unique<DenseLayer<double>>(784, 100))
            .AddLayer(std::make_unique<SigmoidActivationLayer<double>>())
            .AddLayer(std::make_unique<DenseLayer<double>>(100, 10))
            .AddLayer(std::make_unique<SoftmaxActivationLayer<double>>());

    return neural_network;
}

void WritePredictions(const NeuralNetwork<double>& model, const Tensor<double>& x_test, const std::string& output_path) {
    arma::ucolvec predictions = arma::index_max(model.Predict(x_test).Values(), 1);
    std::string predictions_path = output_path + "/predictions.csv";
    nn_framework::io::CsvWriter writer(predictions_path);
    writer.WriteRow({"ImageId", "Label"});
    for (arma::u64 i = 0; i < predictions.n_rows; i++) {
        writer.WriteRow({i + 1, predictions.at(i, 0)});
    }
    std::cout << "Predictions written to " << predictions_path << std::endl;
}

void DigitRecognizer(const std::string &data_path, const std::string &output) {
    auto[x_train, y_train] = LoadMnist(data_path + "/train.csv");
    auto x_test = LoadMnistX(data_path + "/test.csv");

    std::cout << "X: " << FormatDimensions(x_train) << " y: " << FormatDimensions(y_train) << std::endl;
    std::cout << "Start digit-recognizer neural network..." << std::endl;

    auto model = BuildMnistNN();
    model.Fit(x_train, y_train, 40, NoBatches, false, std::make_shared<ProgressBarCallback<double>>());

    auto train_score = nn_framework::scoring::one_hot_accuracy_score(model.Predict(x_train), y_train);
    std::cout << "Final train score: " << train_score << std::endl;

    WritePredictions(model, x_test, output);
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);

    cxxopts::Options options("digit recognizer example");
    options.add_options()("data-path", "path to data", cxxopts::value<std::string>());
    options.add_options()("output-path", "path to save output", cxxopts::value<std::string>());
    auto parsed_args = options.parse(argc, argv);
    auto data_path = parsed_args["data-path"].as<std::string>();
    auto output_path = parsed_args["output-path"].as<std::string>();

    DigitRecognizer(data_path + "/kaggle-digit-recognizer", output_path);
    return 0;
}