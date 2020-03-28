#include <memory>
#include <armadillo>
#include <random>
#include <glog/logging.h>
#include <cxxopts.hpp>
#include <src/io/csv.hpp>
#include <src/io/img.hpp>
#include <src/io/filesystem.hpp>
#include <src/data_processing/data_utils.hpp>
#include <src/scoring/scoring.hpp>
#include "neural_network.hpp"
#include "src/layers/activations.hpp"
#include "optimizer.hpp"
#include "utils.hpp"
#include "layers/dense.hpp"
#include "os_utils.hpp"


void GenerateInputs(arma::mat& inputs, arma::mat& outputs) {
    std::random_device random_device;
    std::mt19937 gen(random_device());
    std::uniform_int_distribution<> coin(0, 1);
    std::uniform_real_distribution real;
    auto inputs_vector = std::vector<std::vector<double>>();
    auto outputs_vector = std::vector<std::vector<double>>();
    for (int i = 0; i < 100; i++) {
        auto type = coin(gen);
        auto radius = 5 * type + 1;
        auto angle = real(gen) * 2 * M_PI;
        auto x = radius * cos(angle) + real(gen) * 0.1;
        auto y = radius * sin(angle) + real(gen) * 0.1;
        inputs_vector.push_back({x, y});
        outputs_vector.push_back({(double) type});
    }
    inputs = CreateMatrix(inputs_vector);
    outputs = CreateMatrix(outputs_vector);
}

void Sample() {
    DLOG(INFO) << "Start example neural network...";
    auto neural_network = NeuralNetwork(std::make_unique<Optimizer>(0.01), std::make_unique<MSELoss>());
    neural_network.AddLayer(std::make_unique<DenseLayer>(2, 3));
    neural_network.AddLayer(std::make_unique<SigmoidActivationLayer>());
    neural_network.AddLayer(std::make_unique<DenseLayer>(3, 1));
    arma::mat inputs, outputs;
    GenerateInputs(inputs, outputs);
    for (int i = 0; i < 10000; i++) {
        std::cout << "Loss: " << neural_network.Fit(inputs, outputs) << std::endl;
    }
    std::cout << arma::join_rows(neural_network.Predict(inputs), outputs) << std::endl;
    std::cout << neural_network.ToString() << std::endl;
}

NeuralNetwork BuildMnistNN(std::unique_ptr<IOptimizer> optimizer) {
    auto neural_network = NeuralNetwork(std::move(optimizer),
                                        std::make_unique<CategoricalCrossEntropyLoss>());
    neural_network.AddLayer(std::make_unique<DenseLayer>(784, 100))
            .AddLayer(std::make_unique<SigmoidActivationLayer>())
            .AddLayer(std::make_unique<DenseLayer>(100, 10))
            .AddLayer(std::make_unique<SoftmaxActivationLayer>());

    return neural_network;
}

void FitNN(NeuralNetwork* neural_network,
           int epochs,
           const arma::mat& x_train,
           const arma::mat& y_train,
           const std::optional<arma::mat>& x_test = std::nullopt,
           const std::optional<arma::mat>& y_test = std::nullopt) {
    Timer timer("Fitting ");
    for (int i = 0; i < epochs; i++) {
        auto loss = neural_network->Fit(x_train, y_train);
        if (i % 5 == 0) {
            auto train_score = nn_framework::scoring::one_hot_accuracy_score(neural_network->Predict(x_train), y_train);
            if (x_test.has_value()) {
                auto test_score = nn_framework::scoring::one_hot_accuracy_score(neural_network->Predict(*x_test),
                                                                                *y_test);
                std::cout << "(" << i << "/" << epochs << ") Loss: " << loss << " Train score: " << train_score
                          << " Test score: " << test_score << std::endl;
            } else {
                std::cout << "(" << i << "/" << epochs << ") Loss: " << loss << " Train score: " << train_score
                          << std::endl;
            }
        } else {
            std::cout << "(" << i << "/" << epochs << ") Loss: " << loss << std::endl;
        }
    }
}

void DigitRecognizer(const std::string& data_path, const std::string& output, std::unique_ptr<IOptimizer> optimizer) {
    auto [x_train, y_train] = LoadMnist(data_path + "/kaggle-digit-recognizer/train.csv", true);
    auto x_test = LoadMnistX(data_path + "/kaggle-digit-recognizer/test.csv", true);

    std::cout << "X: " << FormatDimensions(x_train) << " y: " << FormatDimensions(y_train) << std::endl;
    LOG(INFO) << "Start digit-recognizer neural network...";

    auto neural_network = BuildMnistNN(std::move(optimizer));
    FitNN(&neural_network, 40, x_train, y_train);

    auto train_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_train), y_train);
    std::cout << "Final train score: " << train_score << std::endl;

    arma::ucolvec predictions = arma::index_max(neural_network.Predict(x_test), 1);
    std::string predictions_path = data_path + "/kaggle-digit-recognizer/" + output;
    nn_framework::io::CsvWriter writer(predictions_path);
    writer.WriteRow({"ImageId", "Label"});
    for (arma::u64 i = 0; i < predictions.n_rows; i++) {
        writer.WriteRow({i + 1, predictions.at(i, 0)});
    }
    std::cout << "Predictions written to " << predictions_path << std::endl;
}

void Mnist(const std::string& data_path) {
    auto [x_train, y_train] = LoadMnist(data_path + "/mnist/mnist_train.csv", false);
    auto [x_test, y_test] = LoadMnist(data_path + "/mnist/mnist_test.csv", false);

    std::cout << "X: " << FormatDimensions(x_train) << " y: " << FormatDimensions(y_train) << std::endl;

    LOG(INFO) << "Start mnist neural network...";
    auto neural_network = BuildMnistNN(std::make_unique<Optimizer>(0.00001));
    FitNN(&neural_network, 20, x_train, y_train, x_test, y_test);

    auto train_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_train), y_train);
    auto test_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_test), y_test);
    std::cout << "Final train score: " << train_score << " final test score: " << test_score << std::endl;
}

std::tuple<arma::mat, arma::mat> LoadMnistPng(const std::string& path) {
    std::cout << "Loading png mnist dataset from " << path << std::endl;
    Timer timer("Load of "+ path, true);
    auto paths = nn_framework::io::Filesystem::ListFiles(path);
    nn_framework::io::ImgReader reader(paths);

    arma::uchar_mat x(paths.size(), 28 * 28);
    arma::s32_mat y(paths.size(), 1);
    int pos = 0;
    for (const auto& [name, img] : reader.LoadDataWithNames<arma::u8>()) {
        ensure(img.n_slices == 1, "mnist images should be grayscale");
        x.row(pos) = img.slice(0).as_row();
        y.row(pos) = std::stoi(name.substr(10, 1)); // 000000-numX.png
        pos++;
    }
    auto y_one_hot = nn_framework::data_processing::OneHotEncoding(y);
    return {arma::conv_to<arma::mat>::from(x), arma::conv_to<arma::mat>::from(y_one_hot)};
}

void MnistPng(const std::string& data_path, bool scale) {
    auto [x_train, y_train] = LoadMnistPng(data_path + "/mnist-png/train");
    auto [x_test, y_test] = LoadMnistPng(data_path + "/mnist-png/test");

    if (scale) {
        nn_framework::data_processing::Scaler<double> scaler;
        scaler.Fit(x_train);
        x_train = scaler.Transform(x_train);
        x_test = scaler.Transform(x_test);
    }

    std::cout << "X: " << FormatDimensions(x_train) << " y: " << FormatDimensions(y_train) << std::endl;

    LOG(INFO) << "Start mnist neural network...";
    auto neural_network = BuildMnistNN(std::make_unique<Optimizer>(0.00001));
    FitNN(&neural_network, 20, x_train, y_train, x_test, y_test);

    auto train_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_train), y_train);
    auto test_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_test), y_test);
    std::cout << "Final train score: " << train_score << " final test score: " << test_score << std::endl;
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    cxxopts::Options options("nn framework main");
    options.add_options()
            ("d,data", "path to data", cxxopts::value<std::string>()->default_value("../.."));
    auto parsed_args = options.parse(argc, argv);
    auto data_path = parsed_args["data"].as<std::string>();

    MnistPng(data_path + "/data", false);
    Mnist(data_path + "/data");
    return 0;

    // Mnist(data_path);
//    DigitRecognizer(data_path, "predictions-sgd-0.001.csv", std::make_unique<Optimizer>(0.0001));
//    DigitRecognizer(data_path, "predictions-momentum-0.01.csv", std::make_unique<MomentumOptimizer>(0.0001, 0.0001));
    DigitRecognizer(data_path, "predictions-rmsprop-0.01-40.csv", std::make_unique<RMSPropOptimizer>(0.01));
//    DigitRecognizer(data_path, "predictions-adam-0.01.csv", std::make_unique<AdamOptimizer>(0.01));
//    DigitRecognizer(data_path, "predictions-adam-0.001.csv", std::make_unique<AdamOptimizer>(0.001));
//    DigitRecognizer(data_path, "predictions-adam-0.0001.csv", std::make_unique<AdamOptimizer>(0.0001));
    // Sample();
    return 0;
}