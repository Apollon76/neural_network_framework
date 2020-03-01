#include <memory>
#include <armadillo>
#include <glog/logging.h>
#include "neural_network.hpp"
#include "layers.hpp"
#include "optimizer.hpp"

int main(int, char **argv) {
    google::InitGoogleLogging(argv[0]);
    DLOG(INFO) << "Start example neural network...";
    auto neural_network = NeuralNetwork(std::make_unique<Optimizer>(0.001));
    neural_network.AddLayer(std::make_unique<DenseLayer>(2, 3));
    neural_network.AddLayer(std::make_unique<SigmoidActivationLayer>());
    neural_network.AddLayer(std::make_unique<DenseLayer>(3, 1));
    auto input = arma::mat(
            {
                    {1,  0},
                    {-1, 0},
                    {0,  1},
                    {0,  -1},
                    {3,  0},
                    {-3, 0},
                    {0,  3},
                    {0,  -3}
            });
    auto output = arma::mat({0, 0, 0, 0, 1, 1, 1, 1});
    output.set_size(input.n_rows, 1);
    for (int i = 0; i < 1000; i++) {
        neural_network.Fit(input, output);
    }
    neural_network.Predict(input).print();
    std::cout << neural_network.ToString() << std::endl;
    return 0;
}