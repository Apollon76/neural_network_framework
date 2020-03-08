#include <memory>
#include <armadillo>
#include <random>
#include <glog/logging.h>
#include <src/io/csv.hpp>
#include <src/data_processing/data_utils.hpp>
#include <src/scoring/scoring.hpp>
#include "neural_network.hpp"
#include "layers.hpp"
#include "optimizer.hpp"
#include "utils.hpp"

void GenerateInputs(arma::mat &inputs, arma::mat &outputs) {
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

std::tuple<arma::mat, arma::mat> LoadMnist(const std::string& path) {
    Timer timer("Load of "+ path, true);
    auto csv_data_provider = nn_framework::io::CsvReader(path, true);
    auto data = CreateMatrix(csv_data_provider.LoadData<int>());
    auto X = data.tail_cols(data.n_cols - 1);
    auto y = data.head_cols(1);
    auto y_one_hot = nn_framework::data_processing::OneHotEncoding(y);
    return {arma::conv_to<arma::mat>::from(X), arma::conv_to<arma::mat>::from(y_one_hot)};
}

arma::mat LoadMnistX(const std::string& path) {
    Timer timer("Load of "+ path, true);
    auto csv_data_provider = nn_framework::io::CsvReader(path, true);
    auto data = CreateMatrix(csv_data_provider.LoadData<int>());
    return arma::conv_to<arma::mat>::from(data);
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

NeuralNetwork BuildMnistNN() {
    auto neural_network = NeuralNetwork(std::make_unique<Optimizer>(0.00001), std::make_unique<CategoricalCrossEntropyLoss>());
    neural_network.AddLayer(std::make_unique<DenseLayer>(784, 100));
    neural_network.AddLayer(std::make_unique<SigmoidActivationLayer>());
    neural_network.AddLayer(std::make_unique<DenseLayer>(100, 10));
    neural_network.AddLayer(std::make_unique<SoftmaxActivationLayer>());
    return neural_network;
}

void FitNN(NeuralNetwork *neural_network,
           int epochs,
           const arma::mat &x_train,
           const arma::mat &y_train,
           const std::optional<arma::mat> &x_test=std::nullopt,
           const std::optional<arma::mat> &y_test=std::nullopt) {
    Timer timer("Fitting ");
    for (int i = 0; i < epochs; i++) {
        auto loss = neural_network->Fit(x_train, y_train);
        if (i % 5 == 0) {
            auto train_score = nn_framework::scoring::one_hot_accuracy_score(neural_network->Predict(x_train), y_train);
            if (x_test.has_value()) {
                auto test_score = nn_framework::scoring::one_hot_accuracy_score(neural_network->Predict(*x_test), *y_test);
                std::cout << "(" << i << "/" << epochs << ") Loss: " << loss << " Train score: " << train_score << " Test score: " << test_score << std::endl;
            } else {
                std::cout << "(" << i << "/" << epochs << ") Loss: " << loss << " Train score: " << train_score << std::endl;
            }
        } else {
            std::cout << "(" << i << "/" << epochs << ") Loss: " << loss << std::endl;
        }
    }
}

void DigitRecognizer() {
    auto [x_train, y_train] = LoadMnist("../../data/kaggle-digit-recognizer/train.csv");
    auto x_test = LoadMnistX("../../data/kaggle-digit-recognizer/test.csv");

    std::cout << "X: " << FormatDimensions(x_train) << " y: " << FormatDimensions(y_train) << std::endl;
    LOG(INFO) << "Start digit-recognizer neural network...";

    auto neural_network = BuildMnistNN();
    FitNN(&neural_network, 20, x_train, y_train);

    auto train_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_train), y_train);
    std::cout << "Final train score: " << train_score << std::endl;

    arma::ucolvec predictions = arma::index_max(neural_network.Predict(x_test), 1);
    std::string predictions_path = "../../data/kaggle-digit-recognizer/predictions.csv";
    nn_framework::io::CsvWriter writer(predictions_path);
    writer.WriteRow({"ImageId", "Label"});
    for (arma::u64 i = 0; i < predictions.n_rows; i++) {
        writer.WriteRow({i + 1, predictions.at(i, 0)});
    }
    std::cout << "Predictions written to " << predictions_path << std::endl;
}

void Mnist() {
    auto [x_train, y_train] = LoadMnist("../../data/mnist/mnist_train.csv");
    auto [x_test, y_test] = LoadMnist("../../data/mnist/mnist_test.csv");

    std::cout << "X: " << FormatDimensions(x_train) << " y: " << FormatDimensions(y_train) << std::endl;

    LOG(INFO) << "Start mnist neural network...";
    auto neural_network = BuildMnistNN();
    FitNN(&neural_network, 20, x_train, y_train, x_test, y_test);

    auto train_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_train), y_train);
    auto test_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_test), y_test);
    std::cout << "Final train score: " << train_score << " final test score: " << test_score << std::endl;
}

int main(int, char **argv) {
    google::InitGoogleLogging(argv[0]);
    //Mnist();
    DigitRecognizer();
//    Sample();
}