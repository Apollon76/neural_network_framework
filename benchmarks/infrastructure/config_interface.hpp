#pragma once

#include <tuple>

#include <src/neural_network.hpp>
#include <src/tensor.hpp>

class Config {
public:
    using DataType = std::tuple<Tensor<float>, Tensor<float>, Tensor<float>, Tensor<float>>;

    virtual std::tuple<Tensor<float>, Tensor<float>, Tensor<float>, Tensor<float>> LoadData(const std::string &data_path) = 0;
    virtual NeuralNetwork<float> BuildModel() = 0;
    virtual double GetScore(const Tensor<float> &y_true, const Tensor<float> &y_pred) = 0;
    virtual std::string GetScoreName() = 0;

    virtual ~Config() = default;
};