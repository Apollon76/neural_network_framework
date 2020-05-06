#pragma once

#include "loss.hpp"
#include "tensor.hpp"

template<typename T>
class INeuralNetwork {
public:
    virtual Tensor<T> Predict(const Tensor<T> &input) const = 0;
    virtual ILoss<T>* GetLoss() const = 0;
};