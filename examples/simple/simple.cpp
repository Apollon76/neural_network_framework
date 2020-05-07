#include <src/loss.hpp>
#include <src/tensor.hpp>
#include <src/utils.hpp>
#include <src/optimizer.hpp>
#include <src/layers/dense.hpp>
#include <src/neural_network.hpp>
#include <src/layers/activations.hpp>

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    auto neural_network = NeuralNetwork<double>(
            std::make_unique<RMSPropOptimizer<double>>(0.01),
            std::make_unique<MSELoss<double>>()
    );
    neural_network
            .AddLayer(std::make_unique<DenseLayer<double>>(10, 1))
            .AddLayer(std::make_unique<SoftmaxActivationLayer<double>>());
    Tensor<double> input = Tensor<double>::init(
            {
                    {42, 1, 7},
                    {-5, 6, 666},
                    {0,  9, 9}
            });
    Tensor<double> output = Tensor<double>::init(
            {
                    42 * 5 - 1 * 9 + 7 * 4,
                    -5 * 5 - 6 * 9 + 666 * 4,
                    0 * 5 - 9 * 9 + 9 * 4
            });
    neural_network.Fit(input, output, 10);
    std::cout << dynamic_cast<DenseLayer<double> *>(neural_network.GetLayer(0))->GetWeightsAndBias().Values();
    return 0;
}