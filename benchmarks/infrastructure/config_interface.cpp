#include <tuple>

#include <src/neural_network.hpp>
#include <src/tensor.hpp>

class Config {
public:
    using DataType = std::tuple<Tensor<double>, Tensor<double>, Tensor<double>, Tensor<double>>;

    virtual std::tuple<Tensor<double>, Tensor<double>, Tensor<double>, Tensor<double>> LoadData(const std::string &data_path) = 0;
    virtual NeuralNetwork<double> BuildModel() = 0;
    virtual double GetScore(const Tensor<double> &y_true, const Tensor<double> &y_pred) = 0;
    virtual std::string GetScoreName() = 0;

    virtual ~Config() = default;
};