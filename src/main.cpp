#include <memory>
#include <armadillo>
#include <random>
#include <glog/logging.h>
#include "neural_network.hpp"
#include "layers.hpp"
#include "optimizer.hpp"

arma::mat CreateMatrix(const std::vector<std::vector<double>> &values) {
    auto mat = arma::mat(values.size(), values[0].size());
    for (int i = 0; i < (int) values.size(); i++) {
        for (int s = 0; s < (int) values[0].size(); s++) {
            mat.at(i, s) = values[i][s];
        }
    }
    return mat;
}

void GenerateInputs(arma::mat &inputs, arma::mat &outputs) {
    std::random_device random_device;
    std::mt19937 gen(random_device());
    std::uniform_int_distribution<> coin(0, 1);
    std::uniform_real_distribution real;
    auto inputs_vector = std::vector<std::vector<double>>();
    auto outputs_vector = std::vector<std::vector<double>>();
    for (int i = 0; i < 100; i++) {
        auto type = coin(gen);
        auto radius = 2 * type + 1;
        auto angle = real(gen) * 2 * M_PI;
        auto x = radius * cos(angle) + real(gen) * 0.1;
        auto y = radius * sin(angle) + real(gen) * 0.1;
        inputs_vector.push_back({x, y});
        outputs_vector.push_back({(double) type});
    }
    inputs = CreateMatrix(inputs_vector);
    outputs = CreateMatrix(outputs_vector);
}

int main(int, char **argv) {
    google::InitGoogleLogging(argv[0]);
    DLOG(INFO) << "Start example neural network...";
    auto neural_network = NeuralNetwork(std::make_unique<Optimizer>(0.0001));
    neural_network.AddLayer(std::make_unique<DenseLayer>(2, 3));
    neural_network.AddLayer(std::make_unique<SigmoidActivationLayer>());
    neural_network.AddLayer(std::make_unique<DenseLayer>(3, 1));
    arma::mat inputs, outputs;
    GenerateInputs(inputs, outputs);
    for (int i = 0; i < 10000; i++) {
        neural_network.Fit(inputs, outputs);
        auto loss = neural_network.Predict(inputs);
        loss -= outputs;
        std::cout << "Loss: " << arma::sum(arma::pow(loss, 2));
    }
    return 0;
}