#include <memory>
#include <armadillo>
#include <random>
#include <glog/logging.h>
#include <src/io/csv.hpp>
#include <src/data_processing/data_utils.hpp>
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
    auto csv_data_provider = nn_framework::io::CsvDataProvider(path);
    auto data = CreateMatrix(csv_data_provider.LoadData<int>());
    auto X = data.tail_cols(data.n_cols - 1);
    auto y = data.head_cols(1);
    auto y_one_hot = nn_framework::data_processing::OneHotEncoding(y);
    return {arma::conv_to<arma::mat>::from(X), arma::conv_to<arma::mat>::from(y_one_hot)};
}

void sample() {
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

void mnist() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto [x_train, y_train] = LoadMnist("../../data/mnist_train.csv");
    auto [x_test, y_test] = LoadMnist("../../data/mnist_test.csv");
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Load done in: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

    auto input_sz = x_train.n_cols;
    std::cout << "X: " << FormatDimensions(x_train) << " y: " << FormatDimensions(y_train) << std::endl;

    LOG(INFO) << "Start mnist neural network...";
    auto neural_network = NeuralNetwork(std::make_unique<Optimizer>(0.0001), std::make_unique<CategoricalCrossEntropyLoss>());
    neural_network.AddLayer(std::make_unique<DenseLayer>(input_sz, 100));
    neural_network.AddLayer(std::make_unique<SigmoidActivationLayer>());
    neural_network.AddLayer(std::make_unique<DenseLayer>(100, 10));
    neural_network.AddLayer(std::make_unique<SoftmaxActivationLayer>());
    auto n_iter = 10000;
    for (int i = 0; i < n_iter; i++) {
        auto loss = neural_network.Fit(x_train, y_train);
        if (i % 1 == 0) {
            std::cout << "(" << i << "/" << n_iter << ") Loss: " << loss << std::endl;
        }
    }
    std::cout << arma::join_rows(neural_network.Predict(x_test), y_test) << std::endl;
    std::cout << neural_network.ToString() << std::endl;
}



int main(int, char **argv) {
    google::InitGoogleLogging(argv[0]);
    mnist();
    //sample();
    return 0;
}