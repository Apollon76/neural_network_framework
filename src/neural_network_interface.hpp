#pragma once

#include "tensor.hpp"

template<typename T>
class INeuralNetwork {
public:
    virtual Tensor<T> Predict(const Tensor<T> &input) const = 0;
};