#include <src/loss.hpp>
#include <src/tensor.hpp>
#include <src/utils.hpp>
#include <src/optimizer.hpp>
#include <src/layers/dense.hpp>
#include <src/neural_network.hpp>
#include <src/callbacks/logging_callback.hpp>

int main(int, char **argv) {
    google::InitGoogleLogging(argv[0]);
    auto neural_network = NeuralNetwork<double>(
            std::make_unique<RMSPropOptimizer<double>>(0.01),
            std::make_unique<MSELoss<double>>()
    );
    neural_network.AddLayer(std::make_unique<DenseLayer<double>>(3, 1));
    Tensor<double> input = Tensor<double>::init(
            {
                    {42, 1, 7},
                    {-5, 6, 666},
                    {0,  9, 9},
                    {1,  2, 3},
                    {1,  1, 1},
            });
    Tensor<double> output = Tensor<double>::init(
            {
                    {42 * 5 - 1 * 9 + 7 * 4},
                    {-5 * 5 - 6 * 9 + 666 * 4},
                    {0 * 5 - 9 * 9 + 9 * 4},
                    {1 * 5 - 2 * 9 + 3 * 4},
                    {1 * 5 - 1 * 9 + 1 * 4}
            });
    neural_network.Fit(input, output, 4000, NoBatches, false,
                       EveryNthEpoch<double>(100, std::make_shared<LoggingCallback<double>>())
    );
    std::cout << dynamic_cast<DenseLayer<double> *>(neural_network.GetLayer(0))->GetWeightsAndBias().Values();
    return 0;
}