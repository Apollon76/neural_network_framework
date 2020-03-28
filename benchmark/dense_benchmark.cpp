#include <iostream>
#include <memory>
#include <armadillo>
#include <random>
#include <glog/logging.h>
#include <cxxopts.hpp>
#include "src/io/csv.hpp"
#include "src/data_processing/data_utils.hpp"
#include "src/scoring/scoring.hpp"
#include "src/neural_network.hpp"
#include "src/layers/activations.hpp"
#include "src/optimizer.hpp"
#include "src/utils.hpp"
#include "src/layers/dense.hpp"
#include "src/os_utils.hpp"
using namespace std;

NeuralNetwork BuildDenseNN(std::unique_ptr<IOptimizer> optimizer) {
    auto neural_network = NeuralNetwork(std::move(optimizer),
                                        std::make_unique<CategoricalCrossEntropyLoss>());
    neural_network.AddLayer(std::make_unique<DenseLayer>(784, 100))
            .AddLayer(std::make_unique<SigmoidActivationLayer>())
            .AddLayer(std::make_unique<DenseLayer>(100, 10))
            .AddLayer(std::make_unique<SoftmaxActivationLayer>());

    return neural_network;
}

// TODO: create a special function in some place
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

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);

    cxxopts::Options options("nn framework main");

    options.add_options()
            ("d,test_data", "path to test data", cxxopts::value<std::string>())
            ;
    auto parsed_args = options.parse(argc, argv);
    auto data_path = parsed_args["test_data"].as<std::string>();

    cout << "loading mnist" << endl;

    auto [x_train, y_train] = LoadMnist(data_path + "/train.csv");
    auto [x_test, y_test] = LoadMnist(data_path + "/test.csv");

    cout << "X_train: " << FormatDimensions(x_train) << " y_train: " << FormatDimensions(y_train) << endl;
    cout << "X_test: " << FormatDimensions(x_train) << " y_test: " << FormatDimensions(y_train) << endl;

    auto neural_network = BuildDenseNN(std::make_unique<RMSPropOptimizer>(0.01));
    FitNN(&neural_network, 40, x_train, y_train, x_test, y_test);

    auto test_score = nn_framework::scoring::one_hot_accuracy_score(neural_network.Predict(x_test), y_test);
    cout << "Result on test data: " << test_score << endl;
    return 0;
}